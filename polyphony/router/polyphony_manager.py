import copy
import json
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
        self._pp.umap_transform()
        save and self.save_ann()

    def update_round(self, save=True):
        self._pp.model_update_step()
        save and self.save_ann()

    def get_anchor(self):
        return self._pp.query.anchor

    def put_anchor(self, param):
        anchor = param.get('anchor', None)
        anchor_id = param.get('anchor_id', None)
        operation = param.get('operation', None)  # be either 'add', 'refine', and 'confirm'
        if operation == 'add':
            self._pp.register_anchor(anchor)
        elif operation == 'refine':
            self._pp.refine_anchor(anchor)
        elif operation == 'confirm':
            self._pp.confirm_anchor(anchor_id)
        else:
            raise ValueError("Invalid operation.")
        self.save_anchor_in_json()

    def delete_anchor(self, anchor_id):
        self._pp.delete_anchor(anchor_id)
        self.save_anchor_in_json()

    def save_ann(self, dataset=None):
        dataset = ['query', 'reference'] if dataset is None else dataset
        if 'query' in dataset:
            query = copy.deepcopy(self._pp.query)
            query.anchor = json.dumps(query.anchor)
            query.save_adata(os.path.join(self._folders['zarr'], 'query.zarr'))
            self.save_anchor_in_json()
        if 'reference' in dataset:
            self._pp.ref.save_adata(os.path.join(self._folders['zarr'], 'reference.zarr'))

    def save_anchor_in_json(self):
        file_path = os.path.join(self._folders['json'], 'query_anchor.json')
        with open(file_path, 'w') as f:
            json.dump(self._pp.query.anchor, f)
