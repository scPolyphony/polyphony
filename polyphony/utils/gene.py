from typing import Union, Literal

import numpy as np
import scanpy as sc

from anndata import AnnData


def rank_genes_groups(
    adata: AnnData,
    group_key='anchor_cluster',
    method='wilcoxon'
):
    cls_counts = adata.obs[group_key].value_counts()
    valid_cluster = cls_counts[cls_counts > 1].index.tolist()
    sc.tl.rank_genes_groups(adata, group_key, groups=valid_cluster, method=method)
    adata.uns['rank_genes_groups']['_scores'] = np.array([
        np.array(list(gene_score)) for gene_score in adata.uns['rank_genes_groups']['scores']]).T
    adata.uns['rank_genes_groups']['_names'] = np.array([
        np.array(list(gene_names)) for gene_names in adata.uns['rank_genes_groups']['names']]).T


def get_differential_genes(
    adata: AnnData,
    cluster_idx: str,
    group_key: str = 'anchor_cluster',
    topk: int = 20,
    return_type: Union[Literal['dict', 'matrix']] = 'dict'
):
    cls_counts = adata.obs[group_key].value_counts()
    valid_cluster = cls_counts[cls_counts > 1].index.tolist()
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
