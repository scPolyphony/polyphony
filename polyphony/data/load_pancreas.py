import os

import scanpy as sc
import gdown
from scarches.dataset.trvae.data_handling import remove_sparsity

from polyphony.data import QryAnnDataManager, RefAnnDataManager
from polyphony.utils._constant import DATA_DIR


def load_pancreas(target_conditions=None, data_folder=DATA_DIR):
    condition_key = 'study'
    target_conditions = target_conditions if target_conditions is not None else \
        ['Pancreas inDrop']
    # ['Pancreas CelSeq2', 'Pancreas SS2']
    data_output = os.path.join(data_folder, 'pancreas.h5ad')

    if not os.path.exists(data_output):
        url = 'https://drive.google.com/uc?id=1ehxgfHTsMZXy6YzlFKGJOsBKQ5rrvMnd'
        gdown.download(url, data_output, quiet=False)

    full_adata = sc.read(data_output)
    adata = full_adata.raw.to_adata()
    adata.obs["batch"] = adata.obs[condition_key].tolist()

    sc.pp.filter_cells(adata, min_counts=50)
    sc.pp.filter_cells(adata, min_genes=10)
    adata = remove_sparsity(adata)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    source_adata = adata[~adata.obs[condition_key].isin(target_conditions)].copy()
    target_adata = adata[adata.obs[condition_key].isin(target_conditions)].copy()

    ref_dataset = RefAnnDataManager(source_adata)
    query_dataset = QryAnnDataManager(target_adata)

    return ref_dataset, query_dataset
