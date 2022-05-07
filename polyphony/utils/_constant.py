ANNDATA_REGISTER = dict(
    # keys in anndata.obs
    batch_key='batch',
    cell_type_key='cell_type',
    anchor_assign_key='anchor_cluster',
    anchor_dist_key='anchor_dist',

    # keys in anndata.obsm
    latent_key='latent',
    umap_key='umap',
    anchor_prob_key='anchor_mat',
)

REF_ANNDATA_REGISTER = dict(
    **ANNDATA_REGISTER,
    # keys in anndata.obs
    source_key="source",
)

QRY_ANNDATA_REGISTER = dict(
    **ANNDATA_REGISTER,
    # keys in anndata.obs
    source_key="source",
    label_key="label",
    pred_key="pred",
    pred_prob_key="pred_prob",

    # keys in anndata.uns
    anchor_detail_key="anchor"
)
