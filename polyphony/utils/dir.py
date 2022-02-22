import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

SUPPORTED_ANNDATA_FILETYPE = ['.h5ad', '.zarr']
