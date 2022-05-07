import scanpy as sc

from polyphony.data import QryAnnDataManager, RefAnnDataManager


def umap_transform(
    ref_dataset: RefAnnDataManager,
    qry_dataset: QryAnnDataManager
):
    full_adata = ref_dataset.adata.concatenate(qry_dataset.adata)
    sc.pp.neighbors(full_adata, use_rep='latent')
    sc.tl.umap(full_adata)
    umap = full_adata.obsm['X_umap']
    n_ref = ref_dataset.adata.obs.shape[0]
    ref_dataset.adata.obsm['X_umap'] = umap[:n_ref]
    qry_dataset.adata.obsm['X_umap'] = umap[n_ref:]
