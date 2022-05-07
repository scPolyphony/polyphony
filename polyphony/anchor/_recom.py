from abc import ABC, abstractmethod
from typing import Optional, List

import numpy as np
import pandas as pd
import scanpy as sc
from harmonypy import run_harmony

from ._anchor import Anchor
from polyphony.data import QryAnnDataManager, RefAnnDataManager
from polyphony.tool import rank_gene as rg


class AnchorRecommender(ABC):

    def __init__(
        self,
        ref_dataset: RefAnnDataManager,
        query_dataset: QryAnnDataManager,
        min_count: Optional[int] = 100,
        min_conf: Optional[float] = 0.5,
        clustering_method: Optional[str] = 'leiden',
    ):
        self.qry = query_dataset
        self.ref = ref_dataset

        self._anchor_ref_build_flag = False
        self._min_count = min_count
        self._min_conf = min_conf

        self._clustering_method = clustering_method
        self._anchor_num = 0
        self._param = {}

    @abstractmethod
    def _calc_anchor_assign_prob(self, query_cell_loc=None, *args, **kwargs):
        pass

    def _build_anchor_reference(self):
        if self._anchor_ref_build_flag is False:
            # assign cluster ids to ref cells
            assign = self.ref.anchor_prob.argmax(axis=1).astype('str')
            assign[self.ref.anchor_prob.max(axis=1) < self._min_conf] = 'unsure'
            assign = pd.Series(assign, index=self.ref.adata.obs.index).astype('category')
            self.ref.anchor_assign = assign
            # rank genes according to significance
            rg.rank_genes_groups(self.ref.adata)
            self.ref.adata.uns['_polyphony_params'] = self._param

        self._anchor_ref_build_flag = True

    def recommend_anchors(self, *args, **kwargs) -> List[Anchor]:

        # map query cells to reference clusters
        self._calc_anchor_assign_prob(*args, **kwargs)
        self._build_anchor_reference()
        return self.create_anchors()

    def create_anchors(self) -> List[Anchor]:
        assign_conf = pd.DataFrame(self.qry.anchor_prob, index=self.qry.adata.obs.index)
        anchors = []

        unlabelled = self.qry.adata[self.qry.label == 'none']
        _index = unlabelled.obs.index
        self.qry.anchor_assign = 'none'

        if len(_index) > 0:
            if self._clustering_method == 'leiden':
                sc.pp.neighbors(unlabelled, use_rep='latent')
                sc.tl.leiden(unlabelled)
                self.qry.anchor_assign.loc[_index] = unlabelled.obs['leiden']
            else:
                self.qry.anchor_assign.loc[_index] = assign_conf.loc[_index].argmax(axis=1)

        self.qry.anchor_assign = self.qry.anchor_assign.astype('str')

        local_id_ref = {}

        for anchor_idx in self.qry.anchor_assign.unique():
            if anchor_idx == 'none':
                continue
            self._anchor_num += 1
            cell_index = self.qry.adata.obs[self.qry.anchor_assign == anchor_idx].index
            reference_id = assign_conf.columns[assign_conf.loc[cell_index].sum(axis=0).argmax()]
            anchor_conf = assign_conf.loc[cell_index, reference_id]

            # filter the confident assignment
            valid_cell_index = anchor_conf[anchor_conf > self._min_conf].index

            if len(valid_cell_index) < self._min_count:
                continue

            anchors.append(Anchor(
                id="anchor-{}".format(self._anchor_num),
                reference_id=reference_id,
                cells=[{'cell_id': c, 'dist': 1} for c in valid_cell_index],
                confirmed=False,
                create_by='model'
            ))
            local_id_ref[anchors[-1].id] = anchor_idx

        anchors = self.update_anchors(anchors)

        rg.rank_genes_groups(self.qry.adata)
        for anchor in anchors:
            top_genes = rg.get_differential_genes(self.qry.adata, local_id_ref[anchor.id],
                                                  return_type='matrix')
            anchor.rank_genes_groups = top_genes

        return anchors

    def update_anchors(self, anchors: List[Anchor]):
        assign_conf = pd.DataFrame(self.qry.anchor_assign, index=self.qry.adata.obs.index)
        query_latent = pd.DataFrame(self.qry.latent, index=self.qry.adata.obs.index)
        ref_latent = pd.DataFrame(self.ref.latent, index=self.ref.adata.obs.index)

        for i, anchor in enumerate(anchors):
            cells = [info['cell_id'] for info in anchor.cells]
            # update reference set
            if anchor.create_by == 'user' and not anchor.confirmed:
                anchor.reference_id = int(assign_conf.loc[cells].sum(axis=0).argmax())

            assert anchor.reference_id is not None

            ref_cell_index = self.ref.anchor_assign[
                self.ref.anchor_assign == str(anchor.reference_id)].index
            ref_center = ref_latent.loc[ref_cell_index].mean(axis=0)
            anchor_dist = np.linalg.norm(query_latent.loc[cells] - ref_center, axis=1)

            anchor.cells = [{'cell_id': c, 'dist': float(d)} for c, d in zip(cells, anchor_dist)]
        return anchors


class SymphonyAnchorRecommender(AnchorRecommender):

    def __init__(
        self,
        ref_dataset: RefAnnDataManager,
        query_dataset: QryAnnDataManager,
        n_cluster: Optional[int] = 30,
        sigma: Optional[int] = 0.1,
        **kwargs
    ):
        super(SymphonyAnchorRecommender, self).__init__(ref_dataset, query_dataset, **kwargs)

        self._param.update(
            n_cluster=n_cluster,
            sigma=np.repeat(sigma, n_cluster),
        )

    def _calc_anchor_assign_prob(self, **kwargs):
        if 'reference_cluster_centers' not in self._param.keys():
            hm = run_harmony(
                self.ref.latent,
                self.ref.adata.obs,
                self.ref.batch_key,
                nclust=self._param['n_cluster'],
                sigma=self._param['sigma']
            )
            self._param['reference_cluster_centers'] = hm.Y
            self.ref.anchor_prob = hm.R.T
            self.ref.latent = hm.result().T
            self.ref.anchor_assign = hm.R.argmax(axis=0).T

            self._param['compression_terms'] = dict(
                N=hm.R.sum(axis=1),
                C=np.dot(hm.R, self.ref.latent)
            )

        q_latent = self.qry.latent.T
        normed_q_latent = q_latent / np.linalg.norm(q_latent, ord=2, axis=0)
        normed_center = self._param['reference_cluster_centers'] / \
                        np.linalg.norm(self._param['reference_cluster_centers'], ord=2, axis=0)
        dist_mat = 2 * (1 - np.dot(normed_center.T, normed_q_latent))
        R = -dist_mat
        R = R / self._param['sigma'][:, None]
        R = np.exp(R)
        R = R / np.sum(R, axis=0)
        self.qry.anchor_prob = R.T