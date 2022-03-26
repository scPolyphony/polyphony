import os

import scanpy as sc
import gdown

from scarches.dataset.trvae.data_handling import remove_sparsity

from polyphony.dataset import QueryDataset, ReferenceDataset
from polyphony.utils.dir import DATA_DIR


def load_pbmc(target_conditions=None):
    condition_key = 'study'
    target_conditions = target_conditions if target_conditions is not None else \
        ["10X"]

    data_output = os.path.join(DATA_DIR, 'pbmc.h5ad')

    if not os.path.exists(data_output):
        url = 'https://drive.google.com/uc?id=1Vh6RpYkusbGIZQC8GMFe3OKVDk5PWEpC'
        gdown.download(url, data_output, quiet=False)

    adata = sc.read(data_output)
    adata.X = adata.layers["counts"].copy()
    adata = remove_sparsity(adata)

    sc.pp.filter_cells(adata, min_counts=501)
    sc.pp.filter_cells(adata, max_counts=79534)
    sc.pp.filter_cells(adata, min_genes=54)

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)

    source_adata = adata[~adata.obs[condition_key].isin(target_conditions)].copy()
    target_adata = adata[adata.obs[condition_key].isin(target_conditions)].copy()

    source_adata.raw = source_adata

    sc.pp.highly_variable_genes(
        source_adata,
        n_top_genes=2000,
        batch_key="batch",
        subset=True
    )

    source_adata.X = source_adata.raw[:, source_adata.var_names].X
    target_adata = target_adata[:, source_adata.var_names]

    ref_dataset = ReferenceDataset(source_adata, dataset_id='pbmc_ref',
                                   cell_type_key='final_annotation', batch_key='batch')
    query_dataset = QueryDataset(target_adata, dataset_id='pbmc_query',
                                 cell_type_key='final_annotation', batch_key='batch')

    return ref_dataset, query_dataset
