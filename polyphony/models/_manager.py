import copy

from sklearn.neighbors import KNeighborsClassifier

from polyphony.data import QryAnnDataManager, RefAnnDataManager
from polyphony.models import ActiveSCVI
from polyphony.utils._dir_manager import DirManager


class ModelManager:
    def __init__(
        self,
        instance_id: str,
        ref_dataset: RefAnnDataManager,
        qry_dataset: QryAnnDataManager,
        classifier_cls=KNeighborsClassifier
    ):

        self.ref = ref_dataset
        self.qry = qry_dataset

        self.ref_model = None
        self.qry_model = None
        self.update_qry_models = []
        self.classifier = classifier_cls()
        self.model_iter = 0

        self._dir_manager = DirManager(instance_id=instance_id)

    def setup_anndata(self):
        ActiveSCVI.setup_anndata(self.ref.adata, batch_key=self.ref.batch_key)
        ActiveSCVI.setup_anndata(self.qry.adata, batch_key=self.qry.batch_key)

    def init_reference_step(self, **kwargs):
        self.fit_reference_model(**kwargs)
        self.fit_query_model(**kwargs)
        self.fit_classifier()

    def setup_anndata_anchors(self, confirmed_anchors):
        ActiveSCVI.setup_anchor_rep(self.ref, self.qry, confirmed_anchors=confirmed_anchors)

    def model_update_step(self, **kwargs):
        self.update_query_model(**kwargs)
        self.fit_classifier()

    def fit_classifier(self, transform=True):
        self.classifier.fit(self.ref.latent, self.ref.cell_type)
        if transform:
            self.qry.pred = self.classifier.predict(self.qry.latent)
            self.qry.pred_prob = self.classifier.predict_proba(self.qry.latent)

    def fit_reference_model(self, load_exist=True, save=True, transform=True, **train_kwargs):
        if load_exist and self._dir_manager.model_exists(model_type='ref'):
            self.ref_model = self._dir_manager.load_model(self.ref.adata, model_type='ref')
        else:
            # TODO: move the training parameters to a public function
            self.ref_model = ActiveSCVI(
                self.ref.adata,
                n_layers=2,
                encode_covariates=True,
                deeply_inject_covariates=False,
                use_layer_norm="both",
                use_batch_norm="none",
            )
            self.ref_model.train(max_epochs=600, **train_kwargs)
            save and self._dir_manager.save_model(self.ref_model, 'ref')
        if transform:
            self.ref.latent = self.ref_model.get_latent_representation()

    def fit_query_model(self, load_exist=True, save=True, transform=True, **train_kwargs):
        if load_exist and self._dir_manager.model_exists(
            model_type='qry',
            model_iter=self.model_iter
        ):
            self.qry_model = self._dir_manager.load_model(
                self.qry.adata,
                model_type='qry',
                model_iter=self.model_iter
            )
        else:
            self.qry_model = ActiveSCVI.load_query_data(
                self.qry.adata,
                self._dir_manager.get_model_path('ref'),
                freeze_dropout=True,
            )
            self.qry_model.train(max_epochs=10, **train_kwargs)
            save and self._dir_manager.save_model(self.qry_model, 'qry', self.model_iter)
        if transform:
            self.qry.latent = self.qry_model.get_latent_representation()

    def update_query_model(self, load_exist=False, save=True, transform=True,
                           max_epochs=100, batch_size=256, **train_kwargs):
        self.model_iter += 1
        if load_exist and self._dir_manager.model_exists(
            model_type='qry',
            model_iter=self.model_iter
        ):
            self.ref_model = self._dir_manager.load_model(
                self.qry.adata,
                model_type='qry',
                model_iter=self.model_iter
            )
        else:
            self.update_qry_models.append(copy.deepcopy(self.qry_model))
            self.update_qry_models[-1].train(
                max_epochs=max_epochs,
                early_stopping=True,
                batch_size=batch_size,
                **train_kwargs
            )
            save and self._dir_manager.save_model(
                self.update_qry_models[-1], 'qry', self.model_iter)
        if transform:
            self.qry.latent = self.update_qry_models[-1].get_latent_representation()
