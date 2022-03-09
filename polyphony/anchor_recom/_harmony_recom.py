import numpy as np
import pandas as pd

from harmonypy import run_harmony

from polyphony.anchor_recom import AnchorRecommender


class HarmonyAnchorRecommender(AnchorRecommender):

    def recommend_anchors(self, **kwargs):
        hm = run_harmony(
            np.concatenate([self._ref.latent, self._query.latent]),
            pd.concat([self._ref.obs, self._query.obs]),
            'source',
            max_iter_harmony=0,
            nclust=20,
            **kwargs
        )

        n_ref = len(self._ref.obs)
        self._ref.anchor_mat = hm.R[:, :n_ref].T
        self._query.anchor_mat = hm.R[:, n_ref:].T
        self._ref.anchor_cluster = self._ref.anchor_mat.argmax(axis=1).astype('str')
        self._query.anchor_cluster = self._query.anchor_mat.argmax(axis=1).astype('str')
