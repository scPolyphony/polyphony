import numpy as np

from polyphony.protocol import Protocol


class TopSelectorProtocol(Protocol):
    def __init__(self, anchor_recom=None, call_back=None, anchor_assign=None,
                 top_k=10, sort_by='cluster_distances', ascending=False, *args, **kwargs):

        super(TopSelectorProtocol, self).__init__(anchor_recom, call_back, *args, **kwargs)
        self._anchor_assign = anchor_assign
        self._top_k = top_k
        self._sort_params = dict(by=sort_by, ascending=ascending)

    @property
    def anchor_assign(self):
        return self._anchor_assign

    @property
    def sort_params(self):
        return self._sort_params

    def _update_cluster_stat(self, cluster_stat):
        # TODO: update cluster counts
        return cluster_stat

    @staticmethod
    def filter_valid_cluster(cluster_stat):
        return cluster_stat[(cluster_stat['query_count'] > 0)
                            & (cluster_stat['reference_count'] > 0)]

    def sort_cluster(self, cluster_stat):
        return cluster_stat.sort_values(**self.sort_params)

    def _select_anchors(self, cluster_stat, cluster_assign):
        if self._anchor_assign is None:
            self._anchor_assign = np.zeros(len(cluster_assign))

        cluster_stat = self._update_cluster_stat(cluster_stat)
        cluster_stat = TopSelectorProtocol.filter_valid_cluster(cluster_stat)
        cluster_stat = self.sort_cluster(cluster_stat)
        cluster_ids = cluster_stat.iloc[:self._top_k].index
        cell_update = cluster_assign[cluster_ids, :].sum(axis=0)
        # TODO: update self._anchor_assign
        return cell_update
