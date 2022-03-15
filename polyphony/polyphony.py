"""Main module."""
import copy
import os

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from polyphony.anchor_recom import SymphonyAnchorRecommender
from polyphony.dataset import QueryDataset, ReferenceDataset
from polyphony.models import ActiveSCVI
from polyphony.utils.dir import DATA_DIR
from polyphony.utils.gene import rank_genes_groups, get_differential_genes_by_cell_ids


class Polyphony:
    def __init__(
        self,
        ref_dataset: ReferenceDataset,
        query_dataset: QueryDataset,
        problem_id: str,

        model_cls=ActiveSCVI,
        recommender_cls=SymphonyAnchorRecommender,
        classifier_cls=KNeighborsClassifier,
    ):

        self._ref_dataset = ref_dataset
        self._query_dataset = query_dataset

        self._model_cls = model_cls

        self._ref_model = None
        self._query_model = None
        self._update_query_models = dict()
        self._classifier = classifier_cls()

        self._anchor_recom = recommender_cls(self._ref_dataset, self._query_dataset)
        self._update_id = 0
        self._user_selection_uid = 0

        self._working_dir = os.path.join(DATA_DIR, problem_id)
        self._ref_model_path = os.path.join(self._working_dir, 'model', 'ref_model')
        self._query_model_path = os.path.join(self._working_dir, 'model', 'surgery_model')
        self._umap_model_path = os.path.join(self._working_dir, 'model', 'umap')

        dirs = [os.path.join(self._working_dir, 'model'), os.path.join(self._working_dir, 'data')]
        for dir in dirs:
            os.makedirs(dir, exist_ok=True)

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
        self._model_cls.setup_anndata(self.ref.adata, batch_key=self.ref._batch_key)
        self._model_cls.setup_anndata(self.query.adata, batch_key=self.query._batch_key)

    def init_reference_step(self, **kwargs):
        self._build_reference_latent(**kwargs)
        self._build_query_latent(**kwargs)
        self._fit_classifier()

    def anchor_recom_step(self, **kwargs):
        self._anchor_recom.recommend_anchors(**kwargs)

    def refine_anchor(self, anchor):
        pos = self._find_anchor_by_id(self.query.anchor['unjustified'], anchor['id'])
        anchor = self._anchor_recom.update_anchors([anchor], reassign_ref=False)[0]
        anchor['rank_genes_groups'] = get_differential_genes_by_cell_ids(
            self.query.adata, [c['cell_id'] for c in anchor['cells']])
        self.query.anchor['unjustified'][pos] = anchor

    def label_anchor(self, anchor_id, label):
        raise NotImplementedError

    def register_anchor(self, anchor):
        print(type(anchor['cells']))
        anchor = self._anchor_recom.update_anchors([anchor], reassign_ref=True)[0]
        anchor['rank_genes_groups'] = get_differential_genes_by_cell_ids(self.query.adata,
            [c['cell_id'] for c in anchor['cells']])
        self.query.anchor['user_selection'].append(anchor)

    def delete_anchor(self, anchor_id):
        for group in ['unjustified', 'confirmed', 'user_selection']:
            if anchor_id in [anchor['id'] for anchor in self.query.anchor[group]]:
                pos = self._find_anchor_by_id(self.query.anchor[group], anchor_id)
                self.query.anchor[group].pop(pos)

    def confirm_anchor(self, anchor_id):
        pos = self._find_anchor_by_id(self.query.anchor['unjustified'], anchor_id)
        anchor = self.query.anchor['unjustified'].pop(pos)
        self.query.anchor['confirmed'] = [anchor] + self.query.anchor['confirmed']

    def _find_anchor_by_id(self, anchors, anchor_id):
        pos = next(i for i, anchor in enumerate(anchors) if anchor['id'] == anchor_id)
        return pos

    def _label_step(self, labels, retrain=True):
        self.query.label = labels
        retrain and self._fit_classifier()

    def model_update_step(self, max_epochs=100, batch_size=256, **kwargs):
        self._model_cls.setup_anchor_rep(self.query, self._anchor_recom.compression_terms)
        self._update_id += 1
        self._build_anchored_latent(max_epochs=max_epochs, batch_size=batch_size, **kwargs)
        self._fit_classifier()

    def umap_transform(self, udpate_reference=True, update_query=True):
        udpate_reference and self.ref.umap_transform(dir_name=self._umap_model_path)
        update_query and self.query.umap_transform(model=self.ref.embedder)

    def update_differential_genes(self, **kwargs):
        rank_genes_groups(self.ref.adata, **kwargs)
        rank_genes_groups(self.query.adata, **kwargs)

    def _fit_classifier(self):
        labeled_index = self.query.obs[self.query.label == self.query.label].index  # non-NaN
        X = np.concatenate([self.ref.latent, self.query.latent[labeled_index]])
        y = np.concatenate([self.ref.cell_type, self.query.label[labeled_index]])
        self._classifier.fit(X, y)
        self.query.prediction = self._classifier.predict(self.query.latent)

    def _save_model(self, model_token):
        if model_token == 'ref':
            self._ref_model.save(self._ref_model_path, overwrite=True)
        elif model_token == 'query':
            self._query_model.save(self._query_model_path, overwrite=True)
        else:
            self._query_model.save(os.path.join(self._working_dir, 'model', model_token),
                                   overwrite=True)

    def _load_model(self, model_token):
        if model_token == 'ref':
            self._ref_model = self._model_cls.load(
                dir_path=self._ref_model_path,
                adata=self.ref.adata
            )
        elif model_token == 'query':
            self._query_model = self._model_cls.load(
                dir_path=self._query_model_path,
                adata=self.query.adata
            )
        else:
            self._update_query_models[model_token] = self._model_cls.load(
                dir_path=os.path.join(self._working_dir, model_token),
                adata=self.query.adata
            )

    def _build_ref_model(self, load_exist=True, save=True, **train_kwargs):
        if load_exist and os.path.exists(self._ref_model_path):
            self._load_model('ref')
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
            self._ref_model.train(**train_kwargs)
            save and self._save_model('ref')

    def _build_query_model(self, load_exist=True, save=True, **train_kwargs):
        if load_exist and os.path.exists(self._query_model_path):
            self._load_model('query')
        else:
            self._query_model = self._model_cls.load_query_data(
                self.query.adata,
                self._ref_model_path,
                freeze_dropout=True,
            )
            self._query_model.train(**train_kwargs)
            save and self._save_model('query')

    def _build_reference_latent(self, **kwargs):
        self._build_ref_model(**kwargs)
        self._ref_dataset.latent = self._ref_model.get_latent_representation()

    def _build_query_latent(self, **kwargs):
        self._build_query_model(**kwargs)
        self._query_dataset.latent = self._query_model.get_latent_representation()

    def _build_anchored_latent(self, max_epochs=100, batch_size=512, save=True, **kwargs):
        update_id = "query_iter-{}".format(self._update_id)
        self._update_query_models[update_id] = copy.deepcopy(self._query_model)
        self._update_query_models[update_id].train(
            max_epochs=max_epochs,
            early_stopping=True,
            batch_size=batch_size,
            **kwargs
        )
        save and self._save_model(update_id)
        self._query_dataset.latent = self._update_query_models[update_id] \
            .get_latent_representation()
