import numpy as np

from abc import ABC, abstractmethod
from typing import Optional

from polyphony.dataset import QueryDataset, ReferenceDataset
from polyphony.utils.gene import rank_genes_groups, get_differential_genes


class AnchorRecommender(ABC):

    def __init__(
        self,
        ref_dataset: ReferenceDataset,
        query_dataset: QueryDataset,
        min_count: Optional[int] = 50,
        min_conf: Optional[float] = 0.5,
    ):
        self._query = query_dataset
        self._ref = ref_dataset

        self._anchor_ref_build_flag = False
        self._min_count = min_count
        self._min_conf = min_conf

    def recommend_anchors(self, *args, **kwargs):

        # map query cells to reference clusters
        self._calc_anchor_assign_prob(*args, **kwargs)
        self.build_or_anchor_ref()
        self.build_or_update_anchor()

    def build_or_anchor_ref(self):
        if self._anchor_ref_build_flag is None:
            # assign cluster ids to ref cells
            self._ref.anchor_cluster = self._ref.anchor_mat.argmax(axis=1)
            self._ref.anchor_cluster = self._ref.anchor_cluster.astype('str').astype('category')
            # rank genes according to significance
            rank_genes_groups(self._ref.adata)

        _anchor_ref_build_flag = True

    def build_or_update_anchor(self):

        if self._query.anchor is None:
            self._query.anchor = dict(
                unjustified=[],
                confirmed=[],
                user_selection=[]
            )

        unjustified = []
        assign_conf = self._query.anchor_mat

        # TODO: query.obs['anchor_cluster'] will deprecate soon
        self._query.anchor_cluster = assign_conf.argmax(axis=1)
        self._query.anchor_cluster = self._query.anchor_cluster.astype('str')
        # filter the confident assignment
        self._query.anchor_cluster[assign_conf.max(axis=1) < self._min_conf] = 'unsure'
        self._query.anchor_cluster.astype('category')

        rank_genes_groups(self._query.adata)

        for i in range(self._query.anchor_mat.shape[1]):
            cells = self._query.obs[self._query.anchor_cluster == str(i)].index
            cell_loc = self._query.obs.index.get_indexer_for(cells)
            anchor_dist = assign_conf[cell_loc, i]
            if len(cells) < self._min_count:
                continue
            top_genes = get_differential_genes(self._query.adata, str(i),
                                               topk=self._query.adata.X.shape[1],
                                               return_type='matrix')
            unjustified.append(dict(
                id="cluster-{}".format(i),
                anchor_ref_id=i,
                cells=[{'cell_id': c, 'anchor_dist': d} for c, d in zip(cells, anchor_dist)],
                rank_genes_groups=top_genes,
                top_gene_similarity=1,  # TODO: replace it with the true similarity
                anchor_dist_median=np.median(anchor_dist),
            ))

        self._query.anchor['unjustified'] = unjustified

        for set_name in ['user_selection']:
            for i, anchor in enumerate(self._query.anchor[set_name]):
                cells = [info['cell_id'] for info in anchor['cells']]
                cell_loc = self._query.obs.index.get_indexer_for(cells)
                # update reference set
                anchor['anchor_ref_id'] = assign_conf[cell_loc].sum(axis=0).argmax()
                anchor_dist = assign_conf[cell_loc, anchor['anchor_ref_id']]
                anchor['cells'] = [{'cell_id': c, 'anchor_dist': d}
                                   for c, d in zip(cells, anchor_dist)]
                anchor['anchor_dist_median'] = np.median(anchor_dist)
                # TODO: update top_gene_similarity

    @abstractmethod
    def _calc_anchor_assign_prob(self, query_cell_loc=None, *args, **kwargs):
        pass
