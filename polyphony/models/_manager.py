import copy

from sklearn.neighbors import KNeighborsClassifier

from polyphony.data import QryAnnDataManager, RefAnnDataManager
from polyphony.models import ActiveSCVI


class ModelManager:
    """Manager model operations.

    Args:
        instance_id: str, a unique value to identify the experiment
        ref_dataset: RefAnnDataManager, an object containing the reference dataset with annotations
        qry_dataset: QryAnnDataManager, an object containing the query dataset with annotations
        classifier_cls: Type[Object], the class name of the classifier
    """
    def __init__(
        self,
        instance_id: str,
        ref_dataset: RefAnnDataManager,
        qry_dataset: QryAnnDataManager,
        classifier_cls=KNeighborsClassifier
    ):
        self.instance_id = instance_id
        self.ref = ref_dataset
        self.qry = qry_dataset

        self.ref_model = None
        self.qry_models = []
        self.classifier = classifier_cls()
        self.model_iter = 0

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

    def fit_reference_model(self, transform=True, max_epochs=400, **train_kwargs):
        # TODO: move the training parameters to a public function
        self.ref_model = ActiveSCVI(
            self.ref.adata,
            n_layers=2,
            encode_covariates=True,
            deeply_inject_covariates=False,
            use_layer_norm="both",
            use_batch_norm="none",
        )
        self.ref_model.train(max_epochs=max_epochs, **train_kwargs)
        if transform:
            self.ref.latent = self.ref_model.get_latent_representation()

    def fit_query_model(self, transform=True, max_epochs=10, **train_kwargs):
        self.qry_models.append(ActiveSCVI.load_query_data(
            self.qry.adata,
            self.ref_model,
            freeze_dropout=True,
        ))
        self.qry_models[0].train(max_epochs=max_epochs, **train_kwargs)
        if transform:
            self.qry.latent = self.qry_models[0].get_latent_representation()

    def update_query_model(self, transform=True, max_epochs=100, batch_size=256, **train_kwargs):
        self.model_iter += 1
        self.qry_models.append(copy.deepcopy(self.qry_models[-1]))
        self.qry_models[-1].train(
            max_epochs=max_epochs,
            early_stopping=True,
            batch_size=batch_size,
            **train_kwargs
        )
        if transform:
            self.qry.latent = self.qry_models[-1].get_latent_representation()
