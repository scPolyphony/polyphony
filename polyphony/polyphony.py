"""Main module."""
import copy
import os
import json
from typing import Optional

import anndata
import numpy as np
import scanpy as sc
import torch.cuda
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

from polyphony.anchor_recom import SymphonyAnchorRecommender
from polyphony.benchmark import clisi, f1_lisi, ilisi
from polyphony.dataset import QueryDataset, ReferenceDataset
from polyphony.models import ActiveSCVI
from polyphony.utils.dir import DATA_DIR
from polyphony.utils.gene import get_differential_genes_by_cell_ids


class Polyphony:
    def __init__(
        self,
        ref_dataset: ReferenceDataset,
        query_dataset: QueryDataset,
        instance_id: str,
        iter_id: Optional[int] = None,

        ref_model=None,
        query_model=None,
        update_query_models=None,

        model_cls=ActiveSCVI,
        recommender_cls=SymphonyAnchorRecommender,
        classifier_cls=KNeighborsClassifier,
    ):

        self._ref_dataset = ref_dataset
        self._query_dataset = query_dataset

        self._model_cls = model_cls

        self._ref_model = ref_model
        self._query_model = query_model
        self._update_query_models = [] if update_query_models is None else update_query_models
        self._classifier = classifier_cls()

        self._anchor_recom = recommender_cls(self._ref_dataset, self._query_dataset)
        self._update_id = 0 if iter_id is None else iter_id

        self._working_dir = os.path.join(DATA_DIR, instance_id)
        self._ref_model_path = os.path.join(self._working_dir, 'model', 'ref_model')
        self._query_model_path = os.path.join(self._working_dir, 'model', 'surgery_model')
        self._umap_model_path = os.path.join(self._working_dir, 'model', 'umap')

        os.makedirs(os.path.join(self._working_dir, 'model'), exist_ok=True)
        os.makedirs(os.path.join(self._working_dir, 'data'), exist_ok=True)

    @property
    def ref(self):
        return self._ref_dataset

    @property
    def query(self):
        return self._query_dataset

    @property
    def ref_model(self):
        return self._ref_model

    @ref_model.setter
    def ref_model(self, ref_model):
        self._ref_model = ref_model

    @property
    def query_model(self):
        return self._query_model

    @query_model.setter
    def query_model(self, query_model):
        self._query_model = query_model

    @property
    def full_adata(self):
        return self.ref.adata.concatenate(self.query.adata)

    def setup_data(self):
        self._model_cls.setup_anndata(self.ref.adata, batch_key=self.ref.batch_key)
        self._model_cls.setup_anndata(self.query.adata, batch_key=self.query.batch_key)

    def init_reference_step(self, **kwargs):
        self._build_reference_latent(**kwargs)
        self._build_query_latent(**kwargs)
        self._fit_classifier()

    def anchor_recom_step(self, **kwargs):
        self._anchor_recom.recommend_anchors(**kwargs)

    def model_update_step(self, max_epochs=100, batch_size=256, **kwargs):
        self._model_cls.setup_anchor_rep(self.query, self._anchor_recom.compression_terms)
        self._update_id += 1
        self._build_anchored_latent(max_epochs=max_epochs, batch_size=batch_size, **kwargs)
        self._fit_classifier()

    def refine_anchor(self, anchor):
        pos = self._find_anchor_by_id(self.query.anchor['unjustified'], anchor['id'])
        anchor['anchor_ref_id'] = self.query.anchor['unjustified'][pos]['anchor_ref_id']
        anchor = self._anchor_recom.update_anchors([anchor], reassign_ref=False)[0]
        anchor['rank_genes_groups'] = get_differential_genes_by_cell_ids(
            self.query.adata, [c['cell_id'] for c in anchor['cells']])
        self.query.anchor['unjustified'][pos] = anchor

    def label_anchor(self, anchor_id, label):
        raise NotImplementedError

    def register_anchor(self, anchor):
        anchor = self._anchor_recom.update_anchors([anchor], reassign_ref=True)[0]
        anchor['rank_genes_groups'] = get_differential_genes_by_cell_ids(
            self.query.adata, [c['cell_id'] for c in anchor['cells']])
        self.query.anchor['user_selection'].append(anchor)

    def delete_anchor(self, anchor_id):
        for group in ['unjustified', 'confirmed', 'user_selection']:
            if anchor_id in [anchor['id'] for anchor in self.query.anchor[group]]:
                pos = self._find_anchor_by_id(self.query.anchor[group], anchor_id)
                self.query.anchor[group].pop(pos)

    def confirm_anchor(self, anchor_id):
        for group in ['unjustified', 'user_selection']:
            if anchor_id in [anchor['id'] for anchor in self.query.anchor[group]]:
                pos = self._find_anchor_by_id(self.query.anchor[group], anchor_id)
                anchor = self.query.anchor[group].pop(pos)
                query_cell_ids = [cell['cell_id'] for cell in anchor['cells']]
                self.query.label.loc[query_cell_ids] = self.query.prediction.loc[query_cell_ids] \
                    .cat.set_categories(self.query.label.cat.categories)
                self.query.anchor['confirmed'] = [anchor] + self.query.anchor['confirmed']

    def _find_anchor_by_id(self, anchors, anchor_id):
        pos = next(i for i, anchor in enumerate(anchors) if anchor['id'] == anchor_id)
        return pos

    def umap_transform(self, udpate_reference=True, update_query=True):
        # udpate_reference and self.ref.umap_transform(dir_name=self._umap_model_path)
        # update_query and self.query.umap_transform(model=self.ref.embedder)
        full_adata = self.full_adata
        sc.pp.neighbors(full_adata, use_rep='latent')
        sc.tl.umap(full_adata)
        umap = full_adata.obsm['X_umap']
        n_ref = self.ref.obs.shape[0]
        self.ref.adata.obsm['X_umap'] = umap[:n_ref]
        self.query.adata.obsm['X_umap'] = umap[n_ref:]

    def evaluate(self):
        performance = {
            'ilisi': ilisi([self.ref, self.query]),
            # 'clisi': clisi([self.ref, self.query]),
            # 'f1_lisi': f1_lisi([self.ref, self.query]),
            'accuracy': accuracy_score(self.query.cell_type, self.query.prediction),
            'f1_score': f1_score(self.query.cell_type, self.query.prediction, average='macro'),
        }
        self.query.adata.uns['performance'] = performance

    def _fit_classifier(self):
        labeled_index = self.query.obs[self.query.label != 'none'].index
        # X = np.concatenate([self.ref.latent, self.query.latent[labeled_index]])
        # y = np.concatenate([self.ref.cell_type, self.query.label[labeled_index].values])
        X = self.ref.latent
        y = self.ref.cell_type
        self._classifier.fit(X, y)
        self.query.prediction = self._classifier.predict(self.query.latent)
        self.query.pred_prob = self._classifier.predict_proba(self.query.latent)

    def save_data(self, save_ref=True, save_query=True):
        data_dir = os.path.join(self._working_dir, 'data')
        if save_ref:
            ref_path_prefix = os.path.join(data_dir, 'ref' if self._update_id == 0
                else 'ref_iter-{}'.format(self._update_id))
            self.ref.adata.write(os.path.join(data_dir, ref_path_prefix + '.h5ad'))
        if save_query:
            query = self.query.copy()
            query_path_prefix = os.path.join(data_dir, 'query' if self._update_id == 0
                else 'query_iter-{}'.format(self._update_id))
            if query.anchor is not None:
                query_anchor_str = json.dumps(query.anchor)
                with open(query_path_prefix + '.json', 'w') as f:
                    f.write(query_anchor_str)
                query.anchor = None
            query.adata.write(query_path_prefix + '.h5ad')

    @staticmethod
    def load_data(instance_id, iter=0):
        data_dir = os.path.join(DATA_DIR, instance_id, 'data')
        ref_path_prefix = os.path.join(
            data_dir, 'reference' if iter == 0 else 'reference_iter-{}'.format(iter))
        ref_adata = anndata.read_h5ad(os.path.join(data_dir, ref_path_prefix + '.h5ad'))
        query_path_prefix = os.path.join(
            data_dir, 'query' if iter == 0 else 'query_iter-{}'.format(iter))
        query_adata = anndata.read_h5ad(query_path_prefix + '.h5ad')
        ref = ReferenceDataset(ref_adata)
        query = QueryDataset(query_adata)
        if os.path.exists(query_path_prefix + '.json'):
            with open(query_path_prefix + '.json') as f:
                query.anchor = json.load(f)
        return ref, query

    def _save_model(self, model_token):
        if model_token == 'ref':
            self._ref_model.save(self._ref_model_path, overwrite=True)
        elif model_token == 'query':
            self._query_model.save(self._query_model_path, overwrite=True)
        else:
            self._query_model.save(os.path.join(self._working_dir, 'model', model_token),
                                   overwrite=True)

    def _build_ref_model(self, load_exist=True, save=True, **train_kwargs):
        if load_exist and os.path.exists(self._ref_model_path):
            self._ref_model = self._model_cls.load(
                dir_path=self._ref_model_path,
                adata=self.ref.adata,
                use_gpu=torch.cuda.is_available()
            )
        else:
            # TODO: move the training parameters to a public function
            self._ref_model = self._model_cls(
                self.ref.adata,
                n_layers=2,
                encode_covariates=True,
                deeply_inject_covariates=False,
                use_layer_norm="both",
                use_batch_norm="none",
            )
            self._ref_model.train(use_gpu=torch.cuda.is_available(), max_epochs=600, **train_kwargs)
            save and self._save_model('ref')

    def _build_query_model(self, load_exist=True, save=True, **train_kwargs):
        if load_exist and os.path.exists(self._query_model_path):
            self._query_model = self._model_cls.load(
                dir_path=self._query_model_path,
                adata=self.query.adata,
                use_gpu=torch.cuda.is_available()
            )
        else:
            self._query_model = self._model_cls.load_query_data(
                self.query.adata,
                self._ref_model_path,
                freeze_dropout=True,
            )
            self._query_model.train(
                use_gpu=torch.cuda.is_available(),
                max_epochs=10,
                **train_kwargs)
            save and self._save_model('query')

    def _update_query_model(self, load_exist=False, save=True,
                            max_epochs=100, batch_size=512, **train_kwargs):
        update_id = "query_iter-{}".format(self._update_id)
        model_path = os.path.join(self._working_dir, 'model', update_id)
        if load_exist and os.path.exists(model_path):
            self._update_query_models.append(self._model_cls.load(
                dir_path=os.path.join(self._working_dir, update_id),
                adata=self.query.adata,
                use_gpu=torch.cuda.is_available()
            ))
        else:
            self._update_query_models.append(copy.deepcopy(self._query_model))
            self._update_query_models[-1].train(
                max_epochs=max_epochs,
                early_stopping=True,
                batch_size=batch_size,
                use_gpu=torch.cuda.is_available(),
                **train_kwargs
            )
            save and self._save_model(update_id)

    def _build_reference_latent(self, **kwargs):
        self._build_ref_model(**kwargs)
        self._ref_dataset.latent = self._ref_model.get_latent_representation()

    def _build_query_latent(self, **kwargs):
        self._build_query_model(**kwargs)
        self._query_dataset.latent = self._query_model.get_latent_representation()

    def _build_anchored_latent(self, update=True, **kwargs):
        update and self._update_query_model(**kwargs)
        self._query_dataset.latent = self._update_query_models[-1] \
            .get_latent_representation()
