"""Main module."""
from typing import List, Type

from sklearn.neighbors import KNeighborsClassifier

from polyphony.anchor import Anchor, AnchorRecommender, AnchorSetManager, SymphonyAnchorRecommender
from polyphony.data import RefAnnDataManager, QryAnnDataManager
from polyphony.models import ModelManager
from polyphony.utils._dir_manager import DirManagerMixin


class Polyphony(ModelManager, AnchorSetManager, DirManagerMixin):
    """Polyphony is the topic class of this library.

    Polyphony supports reference-based analysis on single-cell transcriptomics data. The Polyphony
    class inherits attributes from three modules, ModelManger, AnchorSetManager, and DirManager-
    Mixin, which supports users to build and execute scvi models, create and update anchors, and
    save and load intermediate outputs.

    Args:
        instance_id: str, a unique value to identify the experiment
        ref_dataset: RefAnnDataManager, an object containing the reference dataset with annotations
        qry_dataset: QryAnnDataManager, an object containing the query dataset with annotations
        classifier_cls: Type[Object], the class name of the classifier
        recommender_cls: Type[AnchorRecommender], the class name of the anchor recommender
    """
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
