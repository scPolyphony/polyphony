import numpy as np
import pandas as pd

from abc import ABC, abstractmethod

from polyphony.dataset import QueryDataset, ReferenceDataset
from polyphony.utils.math import cluster_agg


class AnchorRecommender(ABC):

    def __init__(
        self,
        ref_dataset: ReferenceDataset,
        query_dataset: QueryDataset
    ):
        self._query = query_dataset
        self._ref = ref_dataset

    def recommend_anchors(self, *args, **kwargs):
        self._recommend_anchors(*args, **kwargs)

        # assign cluster ids to cell and query data
        self._ref.anchor_cluster = self._ref.anchor_mat.argmax(axis=1)
        self._ref.anchor_cluster = self._ref.anchor_cluster.astype('str').astype('category')

        self._query.anchor_cluster = self._query.anchor_mat.argmax(axis=1)
        self._query.anchor_cluster = self._query.anchor_cluster.astype('str').astype('category')

        # calculate the distance of each query cell to its corresponding reference cluster's center
        ref_stat = cluster_agg(self._ref.latent, self._ref.anchor_mat.T)
        ref_mean = ref_stat[ref_stat.columns[::2]].fillna(0)

        anchor_center = np.dot(self._query.anchor_mat[:,  ref_mean.index], ref_mean)
        self._query.anchor_dist = np.linalg.norm(self._query.latent - anchor_center, axis=1)

    @abstractmethod
    def _recommend_anchors(self, *args, **kwargs):
        pass
