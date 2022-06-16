from typing import List, Type, Union

from ._anchor import Anchor
from ._recom import AnchorRecommender, SymphonyAnchorRecommender
from polyphony.data import QryAnnDataManager, RefAnnDataManager
from polyphony.tool import rank_gene as rg


class AnchorSetManager:
    """Manager anchor operations.

    Args:
        ref_dataset: RefAnnDataManager, an object containing the reference dataset with annotations
        qry_dataset: QryAnnDataManager, an object containing the query dataset with annotations
        recommender_cls: Type[AnchorRecommender], the class name of the anchor recommender
    """
    def __init__(
        self,
        ref_dataset: RefAnnDataManager,
        qry_dataset: QryAnnDataManager,
        recommender_cls: Type[AnchorRecommender] = SymphonyAnchorRecommender,
        recommender_params: dict = None,
    ):
        self.ref = ref_dataset
        self.qry = qry_dataset

        if recommender_params is None:
            recommender_params = {}
        self._anchor_recom = recommender_cls(self.ref, self.qry, **recommender_params)
        self.anchors: List[Anchor] = []
        if qry_dataset.anchor is not None:
            self.anchors = [Anchor(**anchor) for anchor in qry_dataset.anchor]

    @property
    def confirmed_anchors(self):
        return [anchor for anchor in self.anchors if anchor.confirmed]

    def _validate_anchor(self, anchor: Union[Anchor, list]):
        pass

    def recommend_anchors(self, *args, **kwargs):
        self.anchors = [anchor for anchor in self.anchors
                        if anchor.confirmed or anchor.create_by == 'user']
        self.anchors += self._anchor_recom.recommend_anchors(*args, **kwargs)

    def join_anchor(self, anchor: Union[Anchor, list]):
        if isinstance(anchor, Anchor):
            anchor = [anchor]
        self._validate_anchor(anchor)
        self.anchors += anchor

    def refine_anchor(self, anchor: Union[Anchor, dict]):
        if isinstance(anchor, dict):
            anchor = Anchor(**anchor)
        pos = find_anchor_by_id(self.anchors, anchor.id)
        if pos >= 0:
            anchor.reference_id = self.anchors[pos].reference_id
            anchor = self._anchor_recom.update_anchors([anchor])[0]
            anchor.rank_genes_groups = rg.get_diff_genes_by_cell_ids(
                self.qry.adata, [c['cell_id'] for c in anchor.cells])
            self.anchors[pos] = anchor

    def register_anchor(self, anchor: Union[Anchor, dict]):
        if isinstance(anchor, dict):
            anchor = Anchor(**anchor)
        anchor = self._anchor_recom.update_anchors([anchor])[0]
        anchor.rank_genes_groups = rg.get_diff_genes_by_cell_ids(
            self.qry.adata, [c['cell_id'] for c in anchor.cells])
        anchor.confirmed = False
        anchor.create_by = 'user'
        self.anchors.append(anchor)

    def delete_anchor(self, anchor_id: str):
        pos = find_anchor_by_id(self.anchors, anchor_id)
        pos >= 0 and self.anchors.pop(pos)

    def confirm_anchor(self, anchor_id: str):
        pos = find_anchor_by_id(self.anchors, anchor_id)
        if pos >= 0:
            self.anchors[pos].confirm()
            # TODO: update the labels of the confirmed cells
            query_cell_ids = [cell['cell_id'] for cell in self.anchors[pos].cells]
            self.qry.label = self.qry.label.astype(str)
            self.qry.label.loc[query_cell_ids] = self.qry.pred.loc[query_cell_ids]


def find_anchor_by_id(anchors: List[Anchor], anchor_id: str):
    try:
        pos = next(i for i, anchor in enumerate(anchors) if anchor.id == anchor_id)
        return pos
    except StopIteration:
        return -1
