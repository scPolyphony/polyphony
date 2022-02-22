import os

import scanpy as sc
import gdown

from scarches.dataset.trvae.data_handling import remove_sparsity

from polyphony.dataset import QueryDataset, ReferenceDataset
from polyphony.utils.dir import DATA_DIR


def load_pancreas(target_conditions=None):
    condition_key = 'study'
    target_conditions = target_conditions if target_conditions is not None else \
        ['Pancreas CelSeq2', 'Pancreas SS2']

    data_output = os.path.join(DATA_DIR, 'pancreas.h5ad')

    if not os.path.exists(data_output):
        url = 'https://drive.google.com/uc?id=1ehxgfHTsMZXy6YzlFKGJOsBKQ5rrvMnd'
        output = 'pancreas.h5ad'
        gdown.download(url, output, quiet=False)
    full_adata = sc.read(data_output)
    adata = full_adata.raw.to_adata()

    adata = remove_sparsity(adata)

    source_adata = adata[~adata.obs[condition_key].isin(target_conditions)].copy()
    target_adata = adata[adata.obs[condition_key].isin(target_conditions)].copy()

    ref_dataset = ReferenceDataset(source_adata, dataset_id='pancreas_ref')
    query_dataset = QueryDataset(target_adata, dataset_id='pancreas_query')

    return ref_dataset, query_dataset
