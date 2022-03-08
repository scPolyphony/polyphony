"""Main module."""
import os

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from polyphony.anchor_recom import HarmonyAnchorRecommender
from polyphony.dataset import QueryDataset, ReferenceDataset
from polyphony.models import ActiveSCVI
from polyphony.utils.dir import DATA_DIR


class Polyphony:
    def __init__(
        self,
        ref_dataset: ReferenceDataset,
        query_dataset: QueryDataset,
        problem_id: str,

        model_cls=ActiveSCVI,
        recommender_cls=HarmonyAnchorRecommender,
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

        self._working_dir = os.path.join(DATA_DIR, problem_id)
        self._ref_model_path = os.path.join(self._working_dir, 'model', 'ref_model')
        self._query_model_path = os.path.join(self._working_dir, 'model', 'surgery_model')

        self._ref_dataset.working_dir = os.path.join(self._working_dir, 'data')
        self._query_dataset.working_dir = os.path.join(self._working_dir, 'data')

        os.makedirs(self._working_dir, exist_ok=True)
        os.makedirs(os.path.join(self._working_dir, 'model'), exist_ok=True)
        os.makedirs(os.path.join(self._working_dir, 'data'), exist_ok=True)

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

    def anchor_recom_step(self):
        self._anchor_recom.recommend_anchors()

    def anchor_update_step(self, query_anchor_mat, **kwargs):
        self._query_dataset.anchor_mat = query_anchor_mat
        self._model_cls.setup_anchor_rep(self.ref, self.query)
        self._update_id += 1
        self._build_anchored_latent(**kwargs)
        self._fit_classifier()

    def umap_transform(self, udpate_reference=True, update_query=True):
        udpate_reference and self.ref.umap_transform()
        update_query and self.query.umap_transform(model=self.ref.embedder)

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

    def _build_ref_model(self, load_exist=True, train=True, save=True, **train_kwargs):
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
            if train:
                self._ref_model.train(**train_kwargs)
                save and self._save_model('ref')

    def _build_query_model(self, load_exist=True, train=True, save=True, **train_kwargs):
        if load_exist and os.path.exists(self._query_model_path):
            self._load_model('query')
        else:
            self._query_model = self._model_cls.load_query_data(
                self.query.adata,
                self._ref_model_path,
                freeze_dropout=True,
            )
            if train:
                self._query_model.train(**train_kwargs)
                save and self._save_model('query')

    def _build_anchored_model(self, update_id='query_update', load_exist=False, train=True,
                              save=True, **train_kwargs):
        if load_exist and os.path.exists(os.path.join(self._working_dir, update_id)):
            self._load_model(update_id)
        else:
            self._update_query_models[update_id] = self._model_cls.load_query_data(
                self.query.adata,
                self._ref_model_path,
                freeze_dropout=True,
            )
            if train:
                self._update_query_models[update_id].train(**train_kwargs)
                save and self._save_model(update_id)

    def _build_reference_latent(self, **kwargs):
        self._build_ref_model(**kwargs)
        self._ref_dataset.latent = self._ref_model.get_latent_representation()

    def _build_query_latent(self, **kwargs):
        self._build_query_model(**kwargs)
        self._query_dataset.latent = self._query_model.get_latent_representation()

    def _build_anchored_latent(self, load_exist=False, **kwargs):
        update_id = str(self._update_id)
        self._build_anchored_model(update_id=update_id, load_exist=load_exist, **kwargs)
        self._query_dataset.latent = self._update_query_models[update_id] \
            .get_latent_representation()
