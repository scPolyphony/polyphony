import numpy as np
import pandas as pd

from harmonypy import run_harmony

from polyphony.anchor_recom import AnchorRecommender


class HarmonyAnchorRecommender(AnchorRecommender):

    def recommend_anchors(self, **kwargs):
        hm = run_harmony(
            np.concatenate([self.ref_latent, self.query_latent]),
            pd.concat([self.ref_obs, self.query_obs]),
            self._source_key,
            **kwargs
        )

        n_ref = len(self.ref_obs)
        self._ref_dataset.anchor_mat = hm.R[:, :n_ref].T
        self._query_dataset.anchor_mat = hm.R[:, n_ref:].T
