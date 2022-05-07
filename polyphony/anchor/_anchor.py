from typing import List, Literal, Optional

import numpy as np


class Anchor:
    def __init__(
        self,
        id: str,
        reference_id: Optional[int] = None,
        confirmed: Optional[bool] = False,
        create_by: Optional[Literal['model', 'user']] = 'model',
        dist_median: Optional[int] = None,
        rank_genes_groups: Optional[dict] = None,

        cells: Optional[List[dict]] = None,
    ):
        self.id = id
        self.reference_id = reference_id
        self.confirmed = confirmed
        self.create_by = create_by
        self.rank_genes_groups = rank_genes_groups

        self._cells = cells
        if dist_median is not None:
            self._dist_median = dist_median
        else:
            self._update_anchor_dist_median()

    def confirm(self):
        if self.confirmed is False:
            raise ValueError("anchor-{} has been confirmed".format(self.id))
        self.confirmed = True

    def _update_anchor_dist_median(self):
        if self.cells is not None:
            anchor_dist = [cell['dist'] for cell in self.cells]
            self._dist_median = float(np.median(anchor_dist))

    @property
    def cells(self):
        return self._cells

    @cells.setter
    def cells(self, cells):
        self._cells = cells
        self._update_anchor_dist_median()

    @property
    def dist_median(self):
        return self._dist_median

    def to_dict(self):
        return dict(
            id=self.id,
            reference_id=self.reference_id,
            cells=self.cells,
            confirmed=self.confirmed,
            create_by=self.create_by,
            dist_median=self.dist_median,
            rank_genes_groups=self.rank_genes_groups
        )
