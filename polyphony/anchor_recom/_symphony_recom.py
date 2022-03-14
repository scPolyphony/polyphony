from typing import Optional

import numpy as np

from harmonypy import run_harmony

from polyphony.anchor_recom import AnchorRecommender
from polyphony.dataset import QueryDataset, ReferenceDataset


class SymphonyAnchorRecommender(AnchorRecommender):

    def __init__(
        self,
        ref_dataset: ReferenceDataset,
        query_dataset: QueryDataset,
        n_cluster: Optional[int] = 100,
        sigma: Optional[int] = 0.1
    ):
        super(SymphonyAnchorRecommender, self).__init__(ref_dataset, query_dataset)

        self._n_cluster = n_cluster
        self._sigma = np.repeat(sigma, n_cluster)
        self._reference_cluster_centers = None
        self._compression_terms = None

    @property
    def reference_cluster_centers(self):
        return self._reference_cluster_centers

    @property
    def compression_terms(self):
        return self._compression_terms

    def _calc_anchor_assign_prob(self, **kwargs):
        if self._reference_cluster_centers is None:
            hm = run_harmony(
                self._ref.latent,
                self._ref.obs,
                self._ref.batch_key,
                nclust=self._n_cluster,
                sigma=self._sigma
            )
            self._reference_cluster_centers = hm.Y
            self._ref.anchor_mat = hm.R.T
            self._compression_terms = dict(N=hm.R.sum(axis=1), C=np.dot(hm.R, self._ref.latent))

        q_latent = self._query.latent.T
        normed_q_latent = q_latent / np.linalg.norm(q_latent, ord=2, axis=0)
        dist_mat = 2 * (1 - np.dot(self._reference_cluster_centers.T, normed_q_latent))
        R = -dist_mat
        R = R / self._sigma[:, None]
        R = np.exp(R)
        R = R / np.sum(R, axis=0)
        self._query.anchor_mat = R.T