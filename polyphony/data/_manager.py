import copy
from uuid import uuid4

from anndata import AnnData

from polyphony.utils._constant import ANNDATA_REGISTER, QRY_ANNDATA_REGISTER, REF_ANNDATA_REGISTER


class AnnDataManager:

    def __init__(
        self,
        adata: AnnData,
        adata_register: dict = None,
    ):
        self.adata = adata
        self.id = str(uuid4())
        self._registry = ANNDATA_REGISTER
        if adata_register is not None:
            self._registry.update(adata_register)
    #     self._register_anndata()
    #
    # def _register_anndata(self, copy=False):
    #     manager = self if copy is False else self.copy()
    #     if self._registry['anchor_prob_key'] not in manager.adata.varm_keys():
    #         manager.anchor_prob = None

    @property
    def latent(self):
        return self.adata.obsm[self._registry["latent_key"]]

    @latent.setter
    def latent(self, latent):
        self.adata.obsm[self._registry["latent_key"]] = latent

    @property
    def batch_key(self):
        return self._registry["batch_key"]

    @property
    def batch(self):
        return self.adata.obs[self._registry["batch_key"]]

    @property
    def cell_type(self):
        return self.adata.obs[self._registry["cell_type_key"]]

    @property
    def anchor_prob(self):
        return self.adata.obsm[self._registry["anchor_prob_key"]]

    @anchor_prob.setter
    def anchor_prob(self, anchor_prob):
        self.adata.obsm[self._registry["anchor_prob_key"]] = anchor_prob

    @property
    def anchor_assign(self):
        return self.adata.obs[self._registry["anchor_assign_key"]]

    @anchor_assign.setter
    def anchor_assign(self, anchor_set):
        self.adata.obs[self._registry["anchor_assign_key"]] = anchor_set

    @property
    def anchor_dist(self):
        return self.adata.obs[self._registry["anchor_dist_key"]]

    @anchor_dist.setter
    def anchor_dist(self, anchor_dist):
        self.adata.obs[self._registry["anchor_dist_key"]] = anchor_dist

    @property
    def umap(self):
        return self.adata.obsm[self._registry["umap_key"]]

    def copy(self):
        return copy.deepcopy(self)


class RefAnnDataManager(AnnDataManager):
    def __init__(
        self,
        adata: AnnData,
        adata_register: dict = None,
    ):
        super(RefAnnDataManager, self).__init__(adata, adata_register)
        self._registry = REF_ANNDATA_REGISTER
        if adata_register is not None:
            self._registry.update(adata_register)
        self._register_anndata()

    def _register_anndata(self, copy=False):
        manager = self if copy is False else self.copy()
        manager.adata.obs[self._registry['source_key']] = 'reference'

    @property
    def source(self):
        return self.adata.obs[self._registry['source_key']]


class QryAnnDataManager(AnnDataManager):
    def __init__(
        self,
        adata: AnnData,
        adata_register: dict = None,
    ):
        super(QryAnnDataManager, self).__init__(adata, adata_register)
        self._registry = QRY_ANNDATA_REGISTER
        if adata_register is not None:
            self._registry.update(adata_register)
        self._register_anndata()

    def _register_anndata(self, copy=False):
        manager = self if copy is False else self.copy()
        manager.adata.obs[self._registry['source_key']] = 'query'
        if self._registry['label_key'] not in manager.adata.var_keys():
            manager.label = 'none'
        if self._registry['pred_key'] not in manager.adata.var_keys():
            manager.pred = None
        if self._registry['pred_prob_key'] not in manager.adata.var_keys():
            manager.pred_prob = None
        if self._registry['anchor_detail_key'] not in manager.adata.uns_keys():
            manager.anchor = None

    @property
    def source(self):
        return self.adata.obs[self._registry['source_key']]

    @property
    def label(self):
        return self.adata.obs[self._registry['label_key']]

    @label.setter
    def label(self, label):
        self.adata.obs[self._registry['label_key']] = label

    @property
    def pred(self):
        return self.adata.obs[self._registry['pred_key']]

    @pred.setter
    def pred(self, pred):
        self.adata.obs[self._registry['pred_key']] = pred

    @property
    def pred_prob(self):
        return self.adata.obs[self._registry['pred_prob_key']]

    @pred_prob.setter
    def pred_prob(self, pred_prob):
        self.adata.obs[self._registry['pred_prob_key']] = pred_prob

    @property
    def anchor(self):
        return self.adata.uns[self._registry['anchor_detail_key']]

    @anchor.setter
    def anchor(self, anchor):
        self.adata.uns[self._registry['anchor_detail_key']] = anchor
