import os

from polyphony import Polyphony
from polyphony.dataset import load_pancreas
from polyphony.router.utils import create_project_folders, SERVER_STATIC_DIR


class PolyphonyManager:
    def __init__(self, problem_id, static_folder=SERVER_STATIC_DIR):
        self._problem_id = problem_id
        if problem_id == 'pancreas_easy':
            self._ref_dataset, self._query_dataset = load_pancreas()
        elif problem_id == 'pancreas_hard':
            self._ref_dataset, self._query_dataset = load_pancreas(
                target_conditions=['Pancreas inDrop'])
        else:
            raise NotImplemented

        self._folders = create_project_folders(problem_id, static_folder)

        self._pp = Polyphony(self._ref_dataset, self._query_dataset, problem_id)
        self._pp.setup_data()
        self._pp.init_reference_step()
        self._pp.umap_transform()

    def init_round(self, save=True):
        self._pp.setup_data()
        self._pp.init_reference_step()
        self._pp.anchor_recom_step()
        anchor_mat = self._pp.query_dataset.anchor_mat > 0.3
        self._pp.anchor_update_step(anchor_mat)
        self._pp.umap_transform()
        save and self.save_ann()

    def save_ann(self, dataset=None):
        dataset = ['query', 'reference'] if dataset is None else dataset
        if 'query' in dataset:
            self._pp.query_dataset.save_adata(os.path.join(self._folders['zarr'], 'query.zarr'))
        if 'reference' in dataset:
            self._pp.ref_dataset.save_adata(os.path.join(self._folders['zarr'], 'reference.zarr'))
