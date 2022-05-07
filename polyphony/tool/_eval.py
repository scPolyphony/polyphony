from typing import List, Literal

import numpy as np
import pandas as pd
from harmonypy import compute_lisi
from sklearn.metrics import accuracy_score, f1_score

from polyphony.data import AnnDataManager, QryAnnDataManager, RefAnnDataManager

agg_fn = {
    'median': np.median,
    'mean': np.mean,
    'max': np.max,
    'min': np.min
}


def ilisi(
    datasets: List[AnnDataManager],
    agg: Literal['median', 'mean'] = 'median',
    perplexity: int = 30
):
    latents = np.concatenate([d.latent for d in datasets])
    obs = pd.concat([d.adata.obs for d in datasets])
    ilisi_dist = compute_lisi(latents, obs, ['source'], perplexity)
    ilisi_agg = agg_fn[agg](ilisi_dist)
    return ilisi_agg


def clisi(
    datasets: List[AnnDataManager],
    agg: Literal['median', 'mean'] = 'median',
    perplexity: int = 30
):
    latents = np.concatenate([d.latent for d in datasets])
    obs = pd.concat([d.adata.obs for d in datasets])
    clisi_dist = compute_lisi(latents, obs, ['cell_type'], perplexity)
    clisi_agg = agg_fn[agg](clisi_dist)
    return clisi_agg


def f1_lisi(
    datasets: List[AnnDataManager],
    perplexity: int = 30
):
    latents = np.concatenate([d.latent for d in datasets])
    obs = pd.concat([d.adata.obs for d in datasets])
    lisi_dist = compute_lisi(latents, obs, ['source', 'cell_type'], perplexity)
    ilisi_dist = lisi_dist[:, 0]
    clisi_dist = lisi_dist[:, 1]
    ilisi_norm = (agg_fn['median'](ilisi_dist) - agg_fn['min'](ilisi_dist)) / \
                 (agg_fn['max'](ilisi_dist) - agg_fn['min'](ilisi_dist))
    clisi_norm = (agg_fn['median'](clisi_dist) - agg_fn['min'](clisi_dist)) / \
                 (agg_fn['max'](clisi_dist) - agg_fn['min'](clisi_dist))
    f1_lisi_norm = 2 * (1 - clisi_norm) * ilisi_norm / (1 - clisi_norm + ilisi_norm)
    return f1_lisi_norm


def evaluate(
    ref_dataset: RefAnnDataManager,
    qry_dataset: QryAnnDataManager
):
    performance = {
        'ilisi': ilisi([ref_dataset, qry_dataset]),
        'clisi': clisi([ref_dataset, qry_dataset]),
        'f1_lisi': f1_lisi([ref_dataset, qry_dataset]),
        'accuracy': accuracy_score(qry_dataset.cell_type, qry_dataset.pred),
        'f1_score': f1_score(qry_dataset.cell_type, qry_dataset.pred, average='macro'),
    }
    qry_dataset.adata.uns['performance'] = performance
