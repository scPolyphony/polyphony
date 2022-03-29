from typing import List, Union, Literal

import numpy as np
import scanpy as sc

from anndata import AnnData


def rank_genes_groups(
    adata: AnnData,
    group_key='anchor_cluster',
    method='wilcoxon',
):
    cls_counts = adata.obs[group_key].value_counts()
    valid_cluster = cls_counts[cls_counts > 1].index.tolist()
    sc.tl.rank_genes_groups(adata, group_key, groups=valid_cluster, method=method, use_raw=False)

    adata.uns['rank_genes_groups']['_scores'] = np.array([
        np.array(list(gene_score)) for gene_score in adata.uns['rank_genes_groups']['scores']]).T
    adata.uns['rank_genes_groups']['_names'] = np.array([
        np.array(list(gene_names)) for gene_names in adata.uns['rank_genes_groups']['names']]).T

    var_index = {name: ind for ind, name in enumerate(adata.var_names.values)}
    names_indices = [[var_index[gene_name] for gene_name in group]
                     for group in adata.uns['rank_genes_groups']['_names']]

    adata.uns['rank_genes_groups']['_names_indices'] = names_indices
    adata.uns['rank_genes_groups']['_valid_cluster'] = np.array(valid_cluster, dtype=np.dtype("|O"))


def get_differential_genes(
    adata: AnnData,
    cluster_idx: str,
    topk: int = 100,
    return_type: Union[Literal['dict', 'matrix']] = 'dict'
):
    valid_cluster = adata.uns['rank_genes_groups']['_valid_cluster'].tolist()
    if cluster_idx not in valid_cluster:
        return []
    else:
        cluster_iidx = valid_cluster.index(cluster_idx)
        scores = adata.uns['rank_genes_groups']['_scores'][cluster_iidx, :topk].tolist()
        names = adata.uns['rank_genes_groups']['_names'][cluster_iidx, :topk].tolist()
        if return_type == 'dict':
            return [dict(name_indice=gene, score=score) for gene, score in zip(names, scores)]
        else:
            return dict(name_indice=names, score=scores)


def get_differential_genes_by_cell_ids(
    adata: AnnData,
    cell_ids: List[str],
    method='wilcoxon',
    topk: int = 100,
    return_type: Union[Literal['dict', 'matrix']] = 'dict'
):
    adata = adata.copy()
    adata.obs['cls'] = 'default'
    adata.obs['cls'].loc[cell_ids] = 'selected'
    rank_genes_groups(adata, group_key='cls', method=method, use_raw=False)
    return get_differential_genes(adata, 'selected', 'cls', topk, return_type)
