"""Main module."""
import os

from polyphony.anchor_recom import HarmonyAnchorRecommender
from polyphony.models import ActiveSCVI
from polyphony.protocol import TopSelectorProtocol
from polyphony.utils.dir import DATA_DIR


class Polyphony:
    def __init__(
        self,
        ref_dataset,
        query_dataset,
        problem_id='example',

        batch_key='batch',
        latent_key='latent',

        model_cls=ActiveSCVI,
        recommender_cls=HarmonyAnchorRecommender,
        # protocol_cls=TopSelectorProtocol,

        working_dir=None
    ):

        self._ref_dataset = ref_dataset
        self._query_dataset = query_dataset

        self._model_cls = model_cls
        self._batch_key = batch_key
        self._latent_key = latent_key

        self._ref_model = None
        self._query_model = None
        self._update_query_models = dict()

        self._anchor_recom = recommender_cls(self._ref_dataset, self._query_dataset)
        # self._protocol = protocol_cls(self._anchor_recommender, self._anchoring_step)
        self._update_id = 0

        self._working_dir = working_dir if working_dir else os.path.join(DATA_DIR, problem_id)
        self._ref_model_path = os.path.join(self._working_dir, 'model', 'ref_model')
        self._query_model_path = os.path.join(self._working_dir, 'model', 'surgery_model')

        self._ref_dataset.working_dir = os.path.join(self._working_dir, 'data')
        self._query_dataset.working_dir = os.path.join(self._working_dir, 'data')

        os.makedirs(self._working_dir, exist_ok=True)
        os.makedirs(os.path.join(self._working_dir, 'model'), exist_ok=True)
        os.makedirs(os.path.join(self._working_dir, 'data'), exist_ok=True)

    @property
    def ref_dataset(self):
        return self._ref_dataset

    @property
    def query_dataset(self):
        return self._query_dataset

    @property
    def reference_adata(self):
        return self._ref_dataset.adata

    @property
    def query_adata(self):
        return self._query_dataset.adata

    @property
    def full_adata(self):
        return self.reference_adata.concatenate(self.query_adata)

    def setup_data(self):
        self._model_cls.setup_anndata(self.reference_adata, batch_key=self._batch_key)
        self._model_cls.setup_anndata(self.query_adata, batch_key=self._batch_key)

    def init_reference_step(self, **kwargs):
        self._build_reference_latent(**kwargs)
        self._build_query_latent(**kwargs)

    def anchor_recom_step(self):
        self._anchor_recom.recommend_anchors()

    def anchor_update_step(self, query_anchor_mat, **kwargs):
        self._query_dataset.anchor_mat = query_anchor_mat
        self._model_cls.setup_anchor_rep(self.ref_dataset, self.query_dataset)
        self._update_id += 1
        self._build_anchored_latent(**kwargs)

    def umap_transform(self, udpate_reference=True, update_query=True):
        udpate_reference and self.ref_dataset.umap_transform()
        update_query and self.query_dataset.umap_transform(model=self.ref_dataset.embedder)

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
                adata=self.reference_adata
            )
        elif model_token == 'query':
            self._query_model = self._model_cls.load(
                dir_path=self._query_model_path,
                adata=self.query_adata
            )
        else:
            self._update_query_models[model_token] = self._model_cls.load(
                dir_path=os.path.join(self._working_dir, model_token),
                adata=self.query_adata
            )

    def _build_ref_model(self, load_exist=True, train=True, save=True, **train_kwargs):
        if load_exist and os.path.exists(self._ref_model_path):
            self._load_model('ref')
        else:
            # TODO: move the training parameters to a public function
            self._ref_model = self._model_cls(
                self.reference_adata,
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
                self.query_adata,
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
                self.query_adata,
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
