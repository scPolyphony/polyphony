from abc import ABC, abstractmethod


class AnchorRecommender(ABC):

    def __init__(self, query_dataset=None, ref_dataset=None, source_key='source'):
        self._query_dataset = query_dataset
        self._ref_dataset = ref_dataset
        self._source_key = source_key

    @property
    def query_latent(self):
        return self._query_dataset.latent

    @property
    def ref_latent(self):
        return self._ref_dataset.latent

    @property
    def query_obs(self):
        return self._query_dataset.obs

    @property
    def ref_obs(self):
        return self._ref_dataset.obs

    @abstractmethod
    def recommend_anchors(self, *args, **kwargs):
        pass
