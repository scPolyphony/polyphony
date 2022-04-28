from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc

from polyphony.data import QryAnnDataManager, RefAnnDataManager
from polyphony.utils.gene import rank_genes_groups, get_differential_genes


class AnchorRecommender(ABC):

    def __init__(
        self,
        ref_dataset: RefAnnDataManager,
        query_dataset: QryAnnDataManager,
        min_count: Optional[int] = 100,
        min_conf: Optional[float] = 0.5,
        clustering_method: Optional[str] = 'leiden',
    ):
        self._query = query_dataset
        self._ref = ref_dataset

        self._anchor_ref_build_flag = False
        self._min_count = min_count
        self._min_conf = min_conf

        self._clustering_method = clustering_method
        self._anchor_num = 0

    def recommend_anchors(self, *args, **kwargs):

        # map query cells to reference clusters
        self._calc_anchor_assign_prob(*args, **kwargs)
        self.build_anchor_ref()
        self.build_or_update_anchor()

    def build_anchor_ref(self):
        if self._anchor_ref_build_flag is False:
            # assign cluster ids to ref cells
            anchor_assign = self._ref.anchor_prob.argmax(axis=1).astype('str')
            anchor_assign[self._ref.anchor_prob.max(axis=1) < self._min_conf] = 'unsure'
            anchor_assign = pd.Series(anchor_assign, index=self._ref.adata.obs.index).astype(
                'category')
            self._ref.anchor_assign = anchor_assign
            # rank genes according to significance
            rank_genes_groups(self._ref.adata)

        _anchor_ref_build_flag = True

    def create_anchors(self):
        assign_conf = pd.DataFrame(self._query.anchor_prob, index=self._query.adata.obs.index)
        anchors = []

        unlabelled = self._query.adata[self._query.label == 'none']
        _index = unlabelled.obs.index
        self._query.anchor_assign = 'none'

        if len(_index) > 0:
            if self._clustering_method == 'leiden':
                sc.pp.neighbors(unlabelled, use_rep='latent')
                sc.tl.leiden(unlabelled)
                self._query.anchor_assign.loc[_index] = unlabelled.obs['leiden']
            else:
                self._query.anchor_assign.loc[_index] = assign_conf.loc[_index].argmax(axis=1)

        self._query.anchor_assign = self._query.anchor_assign.astype('str').astype('category')

        for anchor_idx in self._query.anchor_assign.cat.categories:
            if anchor_idx == 'none':
                continue
            self._anchor_num += 1
            cell_index = self._query.adata.obs[self._query.anchor_assign == anchor_idx].index
            anchor_ref_index = assign_conf.columns[assign_conf.loc[cell_index].sum(axis=0).argmax()]
            anchor_conf = assign_conf.loc[cell_index, anchor_ref_index]

            # filter the confident assignment
            valid_cell_index = anchor_conf[anchor_conf > self._min_conf].index

            if len(valid_cell_index) < self._min_count:
                continue

            anchors.append(dict(
                id="anchor-{}".format(self._anchor_num),
                local_id=anchor_idx,
                anchor_ref_id=anchor_ref_index,
                cells=[{'cell_id': c} for c in valid_cell_index],
                top_gene_similarity=1,  # TODO: replace it with the true similarity
            ))

        anchors = self.update_anchors(anchors, reassign_ref=False)

        rank_genes_groups(self._query.adata)
        for anchor in anchors:
            top_genes = get_differential_genes(self._query.adata, anchor['local_id'],
                                               return_type='matrix')
            anchor['rank_genes_groups'] = top_genes

        return anchors

    def update_anchors(self, anchors, reassign_ref=True):
        assign_conf = pd.DataFrame(self._query.anchor_assign, index=self._query.adata.obs.index)
        query_latent = pd.DataFrame(self._query.latent, index=self._query.adata.obs.index)
        ref_latent = pd.DataFrame(self._ref.latent, index=self._ref.adata.obs.index)

        for i, anchor in enumerate(anchors):
            cells = [info['cell_id'] for info in anchor['cells']]
            # update reference set
            if reassign_ref:
                anchor['anchor_ref_id'] = int(assign_conf.loc[cells].sum(axis=0).argmax())

            assert anchor['anchor_ref_id'] is not None

            ref_cell_index = self._ref.anchor_assign[
                self._ref.anchor_assign == str(anchor['anchor_ref_id'])].index
            ref_center = ref_latent.loc[ref_cell_index].mean(axis=0)
            anchor_dist = np.linalg.norm(query_latent.loc[cells] - ref_center, axis=1)

            anchor['cells'] = [{'cell_id': c, 'anchor_dist': float(d)}
                               for c, d in zip(cells, anchor_dist)]
            anchor['anchor_dist_median'] = float(np.median(anchor_dist))
            # TODO: update top_gene_similarity
        return anchors

    def build_or_update_anchor(self, update_unjustified=True, update_confirmed=True,
                               update_user_selection=True):

        if self._query.anchor is None:
            self._query.anchor = dict(
                unjustified=[],
                confirmed=[],
                user_selection=[]
            )

        if update_unjustified:
            self._query.anchor['unjustified'] = self.create_anchors()

        if update_confirmed:
            self.update_anchors(self._query.anchor['confirmed'], reassign_ref=False)

        if update_user_selection:
            self.update_anchors(self._query.anchor['user_selection'])

    @abstractmethod
    def _calc_anchor_assign_prob(self, query_cell_loc=None, *args, **kwargs):
        pass
