from abc import ABC, abstractmethod


class Protocol(ABC):

    def __init__(self, anchor_recom=None, call_back=None, *args, **kwargs):
        self._am = anchor_recom
        self._call_back = call_back
        self._step = 0

    @abstractmethod
    def _select_anchors(self, *args, **kwargs):
        pass

    def run_step(self):
        _, cluster_assign, cluster_stat = self._am.recommend_anchors()
        query_anchor_latent = self._am.get_anchor_latent(cluster_stat, cluster_assign)
        cell_update = self._select_anchors(cluster_stat, cluster_assign)
        self._call_back(cell_update, query_anchor_latent,
                        "surgery_model_updated_{}".format(self._step))
        self._step += 1
