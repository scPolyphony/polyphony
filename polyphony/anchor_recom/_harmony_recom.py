import numpy as np
import pandas as pd

from harmonypy import run_harmony

from polyphony.anchor_recom import AnchorRecommender


class HarmonyAnchorRecommender(AnchorRecommender):

    def _recommend_anchors(self, **kwargs):
        hm = run_harmony(
            np.concatenate([self._ref.latent, self._query.latent]),
            pd.concat([self._ref.obs, self._query.obs]),
            'source',
            **kwargs
        )

        n_ref = len(self._ref.obs)
        self._ref.anchor_mat = hm.R[:, :n_ref].T
        self._query.anchor_mat = hm.R[:, n_ref:].T
