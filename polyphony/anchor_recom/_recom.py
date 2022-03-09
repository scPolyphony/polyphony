from abc import ABC, abstractmethod
from typing import Optional

from polyphony.dataset import QueryDataset, ReferenceDataset


class AnchorRecommender(ABC):

    def __init__(
        self,
        ref_dataset: Optional[ReferenceDataset] = None,
        query_dataset: Optional[QueryDataset] = None
    ):
        self._query = query_dataset
        self._ref = ref_dataset

    @abstractmethod
    def recommend_anchors(self, *args, **kwargs):
        pass
