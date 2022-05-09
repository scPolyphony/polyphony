import os
import json
from pathlib import Path
from typing import List, Literal

import anndata
import torch
from anndata.compat import OverloadedDict

from polyphony.anchor import Anchor
from polyphony.data import AnnDataManager, RefAnnDataManager, QryAnnDataManager
from polyphony.models import ActiveSCVI
from ._constant import DATA_DIR
from ._json_encoder import NpEncoder


class DirManager:
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
    def __init__(
        self,
        workspace_path=None,
        instance_id=None
    ):
        if workspace_path is None:
            if instance_id is None:
                raise ValueError("Params workspace_path and instance_id cannot both be None.")
            else:
                self._root = os.path.join(DATA_DIR, instance_id)
        else:
            self._root = workspace_path
        os.makedirs(self._root, exist_ok=True)

    @staticmethod
    def _ensure_dir(path: os.PathLike):
        if os.path.isdir(path):
            dir_path = path
        else:
            dir_path = os.path.dirname(path)
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    def get_model_path(
        self,
        model_type: Literal['ref', 'qry', 'cls'],
        model_iter: int = 0
    ) -> os.PathLike:
        path = os.path.join(self._root, 'ref_model') if model_type == 'ref' \
            else os.path.join(self._root, 'iter-%d' % model_iter, model_type + '_model')
        return path

    def save_model(
        self,
        model: ActiveSCVI,
        model_type: Literal['ref', 'qry', 'cls'],
        model_iter: int = 0,
        **kwargs
    ):
        path = self.get_model_path(model_type=model_type, model_iter=model_iter)
        if model_type == 'cls':
            raise NotImplementedError
        else:
            DirManager._ensure_dir(path)
            model.save(path, overwrite=True, **kwargs)

    def model_exists(
        self,
        model_type: Literal['ref', 'qry', 'cls'],
        model_iter: int = 0
    ):
        path = self.get_model_path(model_type=model_type, model_iter=model_iter)
        return os.path.exists(path)

    def load_model(
        self,
        adata: anndata.AnnData,
        model_type: Literal['ref', 'qry', 'cls'],
        model_iter: int = 0,
    ) -> ActiveSCVI:
        dir_path = self.get_model_path(model_type=model_type, model_iter=model_iter)
        if model_type == 'cls':
            raise NotImplementedError
        else:
            model = ActiveSCVI.load(
                dir_path=dir_path,
                adata=adata,
                use_gpu=torch.cuda.is_available()
            )
        return model

    def save_data_uns(
        self,
        uns: OverloadedDict,
        data_type: Literal['ref', 'qry'],
        model_iter: int = 0
    ):
        uns = {key: uns[key] for key in uns.keys()}
        file_path = os.path.join(self._root, 'iter-%d' % model_iter, data_type + '_uns.json')
        DirManager._ensure_dir(file_path)
        with open(file_path, 'w') as f:
            json.dump(uns, f, cls=NpEncoder)

    def data_uns_exists(
        self,
        data_type: Literal['ref', 'qry'],
        model_iter: int = 0
    ):
        file_path = os.path.join(self._root, 'iter-%d' % model_iter, data_type + '_uns.json')
        return os.path.exists(file_path)

    def load_data_uns(
        self,
        data_type: Literal['ref', 'qry'],
        model_iter: int = 0
    ) -> dict:
        file_path = os.path.join(self._root, 'iter-%d' % model_iter, data_type + '_uns.json')
        with open(file_path) as f:
            uns = json.load(f)
        return uns

    def get_data_path(
        self,
        data_type: Literal['ref', 'qry'],
        model_iter: int = 0
    ) -> os.PathLike:
        path = os.path.join(self._root, 'ref.h5ad') if data_type == 'ref' \
            else os.path.join(self._root, 'iter-%d' % model_iter, 'qry.h5ad')
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
        if data_manager.adata.uns is not None:
            self.save_data_uns(dict(data_manager.adata.uns), data_type, model_iter)
        path = self.get_data_path(data_type, model_iter)
        DirManager._ensure_dir(path)
        data_manager.adata.write_h5ad(path)

    def data_exists(
        self,
        data_type: Literal['ref', 'qry'],
        model_iter: int = 0
    ):
        path = self.get_data_path(data_type, model_iter)
        return os.path.exists(path)

    def load_data(
        self,
        data_type: Literal['ref', 'qry'],
        model_iter: int = 0
    ) -> AnnDataManager:
        path = self.get_data_path(data_type, model_iter)
        adata = anndata.read_h5ad(path)
        if self.data_uns_exists(data_type, model_iter):
            adata.uns = self.load_data_uns(data_type, model_iter)
        if data_type == 'ref':
            return RefAnnDataManager(adata)
        else:
            return QryAnnDataManager(adata)
