from abc import ABC, abstractmethod


class AnchorRecommender(ABC):

    def __init__(self, adata_full=None, source_key='source', latent_key='latent'):
        self._adata_full = adata_full
        self._source_key = source_key
        self._latent_key = latent_key

    @property
    def adata_full(self):
        return self._adata_full

    @adata_full.setter
    def adata_full(self, adata_full):
        self._adata_full = adata_full

    @property
    def source_key(self):
        return self._source_key

    @property
    def latent_key(self):
        return self._latent_key

    @abstractmethod
    def recommend_anchors(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_anchor_latent(self, *args, **kwargs):
        pass
