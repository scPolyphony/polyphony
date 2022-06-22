import os
import json
from pathlib import Path
from typing import List, Literal

import anndata
import numpy as np

from polyphony.anchor import Anchor
from polyphony.data import AnnDataManager, RefAnnDataManager, QryAnnDataManager
from polyphony.models import ActiveSCVI
from ._constant import DATA_DIR
from ._json_encoder import NpEncoder


class DirManagerMixin:
    """
    Organize, save, and load the files in the training process.

    workspace_root
    - ref.h5ad
    - qry.h5ad
    - ref_model
    - iter-0
        - qry.h5ad
        - qry_model.h5ad
        - qry_uns.json
        - ref_uns.json
    """

    @property
    def root_dir(self):
        return os.path.join(DATA_DIR, self.instance_id)

    @staticmethod
    def _ensure_dir(path: str):
        if os.path.isdir(path):
            dir_path = path
        else:
            dir_path = os.path.dirname(path)
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_model_path(
        cls,
        instance_id: str,
        model_type: Literal['ref', 'qry', 'cls'],
        model_iter: int = 0
    ) -> str:
        root_dir = os.path.join(DATA_DIR, instance_id)
        path = os.path.join(root_dir, 'ref_model') if model_type == 'ref' \
            else os.path.join(root_dir, 'iter-%d' % model_iter, model_type + '_model')
        return path

    def save_model(
        self,
        model: ActiveSCVI,
        model_type: Literal['ref', 'qry', 'cls'],
        model_iter: int = 0,
        **kwargs
    ):
        path = DirManagerMixin.get_model_path(instance_id=self.instance_id,
                                              model_type=model_type, model_iter=model_iter)
        if model_type == 'cls':
            raise NotImplementedError
        else:
            DirManagerMixin._ensure_dir(path)
            model.save(path, overwrite=True, **kwargs)

    @staticmethod
    def model_exists(
        instance_id: str,
        model_type: Literal['ref', 'qry', 'cls'],
        model_iter: int = 0
    ):
        path = DirManagerMixin.get_model_path(instance_id=instance_id,
                                              model_type=model_type, model_iter=model_iter)
        return os.path.exists(path)

    @classmethod
    def load_model(
        cls,
        instance_id: str,
        adata: anndata.AnnData,
        model_type: Literal['ref', 'qry'],
        model_iter: int = 0,
    ) -> ActiveSCVI:
        dir_path = cls.get_model_path(instance_id, model_type=model_type, model_iter=model_iter)
        model = ActiveSCVI.load(dir_path=dir_path, adata=adata)
        return model

    def save_data_uns(
        self,
        uns: dict,
        data_type: Literal['ref', 'qry'],
        model_iter: int = 0
    ):
        uns = {key: uns[key] for key in uns.keys()}
        file_path = os.path.join(self.root_dir, 'iter-%d' % model_iter, data_type + '_uns.json')
        DirManagerMixin._ensure_dir(file_path)
        with open(file_path, 'w') as f:
            json.dump(uns, f, cls=NpEncoder)

    @staticmethod
    def data_uns_exists(
        instance_id: str,
        data_type: Literal['ref', 'qry'],
        model_iter: int = 0
    ):
        file_path = os.path.join(DATA_DIR, instance_id, 'iter-%d' % model_iter,
                                 data_type + '_uns.json')
        return os.path.exists(file_path)

    @staticmethod
    def load_data_uns(
        instance_id: str,
        data_type: Literal['ref', 'qry'],
        model_iter: int = 0
    ) -> dict:
        file_path = os.path.join(DATA_DIR, instance_id, 'iter-%d' % model_iter,
                                 data_type + '_uns.json')
        with open(file_path) as f:
            uns = json.load(f)

        # TODO: change the types to zarr.js' supporting ones
        if data_type == 'ref' and 'rank_genes_groups' in uns.keys():
            uns['rank_genes_groups']['_names_indices'] = np.array(
                uns['rank_genes_groups']['_names_indices'], dtype=np.dtype("uint16"))
            uns['rank_genes_groups']['_valid_cluster'] = np.array(
                uns['rank_genes_groups']['_valid_cluster'], dtype=np.dtype("|O"))
        return uns

    @classmethod
    def get_data_path(
        cls,
        instance_id: str,
        data_type: Literal['ref', 'qry'],
        model_iter: int = 0
    ) -> str:
        root_dir = os.path.join(DATA_DIR, instance_id)
        path = os.path.join(root_dir, 'ref.h5ad') if data_type == 'ref' \
            else os.path.join(root_dir, 'iter-%d' % model_iter, 'qry.h5ad')
        return path

    def save_data(
        self,
        data_manager: AnnDataManager,
        data_type: Literal['ref', 'qry'],
        model_iter: int = 0,
        anchors: List[Anchor] = None
    ):
        if anchors is not None:
            data_manager.anchor = [anchor.to_dict() for anchor in anchors]
        # if data_manager.adata.uns is not None:
        #     self.save_data_uns(dict(data_manager.adata.uns), data_type, model_iter)
        path = DirManagerMixin.get_data_path(self.instance_id, data_type, model_iter)
        DirManagerMixin._ensure_dir(path)
        data_manager.adata.write_h5ad(path)

    @classmethod
    def data_exists(
        cls,
        instance_id: str,
        data_type: Literal['ref', 'qry'],
        model_iter: int = 0
    ):
        path = cls.get_data_path(instance_id, data_type, model_iter)
        return os.path.exists(path)

    @classmethod
    def load_data(
        cls,
        instance_id: str,
        data_type: Literal['ref', 'qry'],
        model_iter: int = 0
    ) -> AnnDataManager:
        path = cls.get_data_path(instance_id, data_type, model_iter)
        adata = anndata.read_h5ad(path)
        # if cls.data_uns_exists(instance_id, data_type, model_iter):
        #     adata.uns = cls.load_data_uns(instance_id, data_type, model_iter)
        if data_type == 'ref':
            return RefAnnDataManager(adata)
        else:
            return QryAnnDataManager(adata)

    def save_snapshot(self):
        self.save_model(self.ref_model, 'ref')
        self.save_model(self.qry_model, 'qry', self.model_iter)

        self.save_data(self.ref, 'ref', self.model_iter)
        self.save_data(self.qry, 'qry', self.model_iter)

    @classmethod
    def load_snapshot(
        cls,
        instance_id: str,
        model_iter: int = 0,
        **kwargs,
    ):
        ref = cls.load_data(instance_id, 'ref', model_iter)
        qry = cls.load_data(instance_id, 'qry', model_iter)
        manager = cls(instance_id, ref, qry, **kwargs)
        manager.ref_model = cls.load_model(instance_id, ref.adata, 'ref')
        manager.qry_model = cls.load_model(instance_id, qry.adata, 'qry', model_iter)
        return manager
