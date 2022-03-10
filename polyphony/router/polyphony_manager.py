import json
import os

import pandas as pd

from polyphony import Polyphony
from polyphony.dataset import load_pancreas
from polyphony.router.utils import create_project_folders, SERVER_STATIC_DIR
from polyphony.utils.gene import get_differential_genes


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
        self._pp.update_differential_genes()
        save and self.save_ann()

    def update_round(self, param, save=True):
        anchors = param.get('anchors', [])

        labels = pd.Series(index=self._pp.query.index)
        ref_anchor_mat = pd.DataFrame(index=self._pp.query.index)
        query_anchor_mat = pd.DataFrame(index=self._pp.query.index)
        for i, anchor in enumerate(anchors):
            anchor_id = anchor.get('id', str(i))
            ref_index = anchor.get('ref_index', [])
            ref_anchor_mat[anchor_id] = 0
            ref_anchor_mat.loc[ref_index, anchor_id] = 1

            query_index = anchor.get('query_index', [])
            query_anchor_mat[anchor_id] = 0
            query_anchor_mat.loc[query_index, anchor_id] = 1
            labels.loc[query_index] = self._pp.query.labels.loc[query_index]

        self._pp.anchor_update_step(ref_anchor_mat, query_anchor_mat)
        self._pp.label_step(labels)

        self._pp.umap_transform()
        self._pp.update_differential_genes()

        save and self.save_ann()

    def anchor_to_json(self, save=False):
        ref_anchor_cluster = self._pp.ref.anchor_cluster
        query_anchor_cluster = self._pp.query.anchor_cluster

        ref_cluster_index = ref_anchor_cluster.cat.categories.tolist()
        query_cluster_index = query_anchor_cluster.cat.categories.tolist()
        cluster_index = [idx for idx in query_cluster_index if idx in ref_cluster_index]
        clusters = []
        for idx in cluster_index:
            ref_index = ref_anchor_cluster[ref_anchor_cluster == idx].index
            query_index = query_anchor_cluster[query_anchor_cluster == idx].index
            query_info = pd.concat([self._pp.query.anchor_dist, self._pp.query.prediction]).to_dict()
            differential_genes = dict(
                ref=get_differential_genes(self._pp.ref.adata, idx),
                query=get_differential_genes(self._pp.query.adata, idx)
            )
            clusters.append(dict(
                id=idx,
                ref_index=ref_index.tolist(),
                query_index=query_index.tolist(),
                query_info=query_info,
                differential_genes=differential_genes
            ))
        if save:
            file_path = os.path.join(self._folders['json'], 'anchor_{}'.format(self._pp._update_id))
            with open(file_path, 'w') as f:
                json.dump(clusters, f)
            return os.path.relpath(file_path, SERVER_STATIC_DIR)
        else:
            return clusters

    def save_ann(self, dataset=None):
        dataset = ['query', 'reference'] if dataset is None else dataset
        if 'query' in dataset:
            self._pp.query.save_adata(os.path.join(self._folders['zarr'], 'query.zarr'))
        if 'reference' in dataset:
            self._pp.ref.save_adata(os.path.join(self._folders['zarr'], 'reference.zarr'))
