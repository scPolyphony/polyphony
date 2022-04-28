"""Main module."""
import copy
import os
from typing import Optional

import scanpy as sc
import torch.cuda
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

from polyphony.anchor_recom import SymphonyAnchorRecommender
from polyphony.benchmark import clisi, f1_lisi, ilisi
from polyphony.data import QryAnnDataManager, RefAnnDataManager
from polyphony.dir import DirManager
from polyphony.models import ActiveSCVI
from polyphony.utils._constant import DATA_DIR
from polyphony.utils.gene import get_differential_genes_by_cell_ids


class Polyphony:
    def __init__(
        self,
        ref_dataset: RefAnnDataManager,
        query_dataset: QryAnnDataManager,
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

        self.ref_model = ref_model
        self.query_model = query_model
        self._update_query_models = [] if update_query_models is None else update_query_models
        self._classifier = classifier_cls()

        self._anchor_recom = recommender_cls(self._ref_dataset, self._query_dataset)
        self._model_iter = 0 if iter_id is None else iter_id

        self._dir_manager = DirManager(os.path.join(DATA_DIR, instance_id))

    @property
    def ref(self):
        return self._ref_dataset

    @property
    def query(self):
        return self._query_dataset

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
        self._model_iter += 1
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
                self.query.label.loc[query_cell_ids] = self.query.pred.loc[query_cell_ids]
                self.query.anchor['confirmed'] = [anchor] + self.query.anchor['confirmed']

    def _find_anchor_by_id(self, anchors, anchor_id):
        pos = next(i for i, anchor in enumerate(anchors) if anchor['id'] == anchor_id)
        return pos

    def umap_transform(self):
        full_adata = self.full_adata
        sc.pp.neighbors(full_adata, use_rep='latent')
        sc.tl.umap(full_adata)
        umap = full_adata.obsm['X_umap']
        n_ref = self.ref.adata.obs.shape[0]
        self.ref.adata.obsm['X_umap'] = umap[:n_ref]
        self.query.adata.obsm['X_umap'] = umap[n_ref:]

    def evaluate(self):
        performance = {
            'ilisi': ilisi([self.ref, self.query]),
            'clisi': clisi([self.ref, self.query]),
            'f1_lisi': f1_lisi([self.ref, self.query]),
            'accuracy': accuracy_score(self.query.cell_type, self.query.pred),
            'f1_score': f1_score(self.query.cell_type, self.query.pred, average='macro'),
        }
        self.query.adata.uns['performance'] = performance

    def _fit_classifier(self):
        X = self.ref.latent
        y = self.ref.cell_type
        self._classifier.fit(X, y)
        self.query.pred = self._classifier.predict(self.query.latent)
        self.query.pred_prob = self._classifier.predict_proba(self.query.latent)

    def _build_ref_model(self, load_exist=True, save=True, **train_kwargs):
        if load_exist and self._dir_manager.model_exists(model_type='ref'):
            self._ref_model = self._dir_manager.load_model(self.ref.adata, model_type='ref')
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
            save and self._dir_manager.save_model(self._ref_model, 'ref')

    def _build_query_model(self, load_exist=True, save=True, **train_kwargs):
        if load_exist and self._dir_manager.model_exists(
            model_type='qry',
            model_iter=self._model_iter
        ):
            self._query_model = self._dir_manager.load_model(
                self.query.adata,
                model_type='qry',
                model_iter=self._model_iter
            )
        else:
            self._query_model = self._model_cls.load_query_data(
                self.query.adata,
                self._dir_manager.get_model_path('ref'),
                freeze_dropout=True,
            )
            self._query_model.train(
                use_gpu=torch.cuda.is_available(),
                max_epochs=10,
                **train_kwargs)
            save and self._dir_manager.save_model(self._query_model, 'qry', self._model_iter)

    def _update_query_model(self, load_exist=False, save=True,
                            max_epochs=100, batch_size=512, **train_kwargs):
        if load_exist and self._dir_manager.model_exists(
            model_type='qry',
            model_iter=self._model_iter
        ):
            self._ref_model = self._dir_manager.load_model(
                self.ref.adata,
                model_type='qry',
                model_iter=self._model_iter
            )
        else:
            self._update_query_models.append(copy.deepcopy(self._query_model))
            self._update_query_models[-1].train(
                max_epochs=max_epochs,
                early_stopping=True,
                batch_size=batch_size,
                use_gpu=torch.cuda.is_available(),
                **train_kwargs
            )
            save and self._dir_manager.save_model(
                self._update_query_models[-1], 'qry', self._model_iter)

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
