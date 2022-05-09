"""Main module."""
from typing import List, Type

from sklearn.neighbors import KNeighborsClassifier

from polyphony.anchor import Anchor, AnchorRecommender, AnchorSetManager, SymphonyAnchorRecommender
from polyphony.data import RefAnnDataManager, QryAnnDataManager
from polyphony.models import ModelManager
from polyphony.utils._dir_manager import DirManagerMixin


class Polyphony(ModelManager, AnchorSetManager, DirManagerMixin):
    def __init__(
        self,
        instance_id: str,
        ref_dataset: RefAnnDataManager,
        qry_dataset: QryAnnDataManager,
        classifier_cls=KNeighborsClassifier,
        recommender_cls: Type[AnchorRecommender] = SymphonyAnchorRecommender,
    ):

        super().__init__(instance_id, ref_dataset, qry_dataset, classifier_cls)
        self._anchor_recom = recommender_cls(self.ref, self.qry)
        self.anchors: List[Anchor] = []
        if qry_dataset.anchor is not None:
            self.anchors = [Anchor(**anchor) for anchor in qry_dataset.anchor]
