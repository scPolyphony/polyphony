import numpy as np
import pandas as pd

from harmonypy import run_harmony

from polyphony.anchor_recom import AnchorRecommender


class HarmonyAnchorRecommender(AnchorRecommender):
    def __init__(self, adata_full=None, conf_threshold=0.3, *args, **kwargs):
        super(HarmonyAnchorRecommender, self).__init__(adata_full, *args, **kwargs)
        self._conf_threshold = conf_threshold

    @property
    def conf_threshold(self):
        return self._conf_threshold

    @property
    def query_latent(self):
        return self.adata_full[self.adata_full.obs[self.source_key] == 'query']\
            .obsm[self.latent_key]

    @property
    def ref_latent(self):
        return self.adata_full[self.adata_full.obs[self.source_key] == 'reference']\
            .obsm[self.latent_key]

    def recommend_anchors(self, return_stat=True):

        hm = run_harmony(
            self.adata_full.obsm[self.latent_key],
            self.adata_full.obs,
            self.source_key
        )

        cluster_assignment = np.zeros(hm.R.shape)
        for row, col in enumerate(hm.R.argmax(axis=0)):
            cluster_assignment[col, row] = 1
        cluster_assignment[:, (hm.R > self.conf_threshold).sum(axis=0) == 0] = 0
        full_obs = self.adata_full.obs.reset_index(drop=True)
        ref_assign = cluster_assignment[:, full_obs[full_obs['source'] == 'reference'].index]
        query_assign = cluster_assignment[:, full_obs[full_obs['source'] == 'query'].index]

        if return_stat:
            count = cluster_assignment.sum(axis=1)

            query_count = query_assign.sum(axis=1)
            ref_count = ref_assign.sum(axis=1)

            query_centers = np.dot(query_assign, self.query_latent) / query_count[:, None]
            ref_centers = np.dot(ref_assign, self.ref_latent) / ref_count[:, None]
            cluster_distances = np.linalg.norm(query_centers - ref_centers, axis=1)

            cluster_stat = pd.DataFrame(dict(
                count=count,
                query_count=query_count,
                reference_count=ref_count,
                cluster_distances=cluster_distances,
                query_centers=query_centers.tolist(),
                ref_centers=ref_centers.tolist()
            ))
            return ref_assign, query_assign, cluster_stat
        return ref_assign, query_assign

    def get_anchor_latent(self, cluster_stat, cluster_assign):
        bias_mat = np.stack(cluster_stat['ref_centers'].values) - \
                   np.stack(cluster_stat['query_centers'].values)
        bias_mat[bias_mat != bias_mat] = 0
        fix = np.dot(cluster_assign.T, bias_mat)
        return np.array(self.query_latent) + fix
