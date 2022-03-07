import copy
import json
import os
import numpy as np

import anndata

from scarches.dataset.trvae.data_handling import remove_sparsity
from umap.parametric_umap import ParametricUMAP, load_ParametricUMAP

from polyphony.utils.dir import DATA_DIR, SUPPORTED_ANNDATA_FILETYPE


class Dataset:

    def __init__(
        self,
        adata,
        dataset_id='dataset',
        batch_key='batch',
        latent_key='latent',
        anchor_key='anchor_cluster',
        working_dir=DATA_DIR
    ):
        self._adata = adata
        self._dataset_id = dataset_id

        self._batch_key = batch_key
        self._latent_key = latent_key
        self._anchor_key = anchor_key

        self._working_dir = working_dir

        self._embedder = None

    @property
    def adata(self):
        return self._adata

    @adata.setter
    def adata(self, adata):
        self._adata = adata

    @property
    def obs(self):
        return self._adata.obs

    @property
    def obsm(self):
        return self._adata.obsm

    @property
    def X(self):
        return self._adata.X

    @property
    def latent(self):
        return self._adata.obsm[self._latent_key]

    @latent.setter
    def latent(self, latent):
        self._adata.obsm[self._latent_key] = latent

    @property
    def batch(self):
        return self._adata.obs[self._batch_key]

    @property
    def anchor_mat(self):
        return self._adata.obsm[self._anchor_key]

    @anchor_mat.setter
    def anchor_mat(self, anchor_mat):
        self._adata.obsm[self._anchor_key] = anchor_mat.astype(np.dtype('<f4'))

    @property
    def umap(self):
        return self._adata.obsm['umap']

    @property
    def embedder(self):
        return self._embedder

    def copy(self):
        return copy.deepcopy(self)

    def preprocess(self, inplace=True):
        dataset = self if inplace else self.copy()
        dataset.adata = remove_sparsity(dataset.adata)
        return dataset

    def _get_umap_input(self, adata, source='latent'):
        if source == 'latent':
            umap_input = adata.obsm[self._latent_key]
        elif source == 'raw':
            umap_input = adata.X
        else:
            raise ValueError("Unsupported umap source.")
        return umap_input

    def _load_umap_model(self, embedder_path):
        if os.path.exists(embedder_path):
            self._embedder = load_ParametricUMAP(embedder_path)

    def _save_umap_model(self, embedder_path):
        if self._embedder is not None:
            self._embedder.save(embedder_path)

    def build_umap_model(self, adata=None, source='latent', load_exist=True, save=True,
                         **train_kwargs):
        adata = self.adata if adata is None else adata
        embedder_path = os.path.join(self._working_dir, 'umap')
        if load_exist and os.path.exists(embedder_path):
            self._load_umap_model(embedder_path)
        else:
            self._embedder = ParametricUMAP()
            self._embedder.fit(self._get_umap_input(adata, source), **train_kwargs)
            save and self._save_umap_model(embedder_path)

    def umap_transform(self, model=None, source='latent', inplace=True):
        dataset = self if inplace else self.copy()
        if model is None and self.embedder is None:
            self.build_umap_model(dataset.adata, source=source)
        model = self.embedder if model is None else model

        umap_input = self._get_umap_input(dataset.adata, source)
        dataset.adata.obsm['X_umap'] = model.transform(umap_input)

    def save_adata(self, path):
        _, file_extension = os.path.splitext(path)
        if file_extension not in SUPPORTED_ANNDATA_FILETYPE:
            raise ValueError("Unsupported AnnData file extension.")
        if file_extension == '.zarr':
            self.adata.write_zarr(path)
        elif file_extension == '.h5ad':
            self.adata.write(path)

    def save(self):
        self._save_umap_model(os.path.join(self._working_dir, 'umap'))
        # TODO: support more file extension in the future
        self.save_adata(os.path.join(self._working_dir, self._dataset_id + '.h5ad'))
        config = dict(
            dataset_id=self._dataset_id,
            batch_key=self._batch_key,
            latent_key=self._latent_key,
            working_dir=self._working_dir
        )
        with open(os.path.join(self._working_dir, '{}_config.json'.format(self._dataset_id)),
                  'w') as f:
            json.dump(config, f)

    @staticmethod
    def load_adata(file_path=None, file_extension='.h5ad'):
        assert file_extension == '.h5ad'
        # TODO: support more file extension in the future
        return anndata.read_h5ad(file_path)

    @classmethod
    def load(cls, dir_path, dataset_id):
        with open(os.path.join(dir_path, '{}_config.json'.format(dataset_id))) as f:
            config = json.load(f)
        adata_file_path = os.path.join(config['working_dir'], '{}.h5ad'.format(dataset_id))
        embedder_path = os.path.join(config['working_dir'], 'umap')
        adata = cls.load_adata(adata_file_path)
        dataset = cls(adata=adata, **config)
        dataset._load_umap_model(embedder_path)
        return dataset
