import numpy as np
import pandas as pd

from typing import List, Literal

from harmonypy import compute_lisi

from polyphony.dataset import Dataset

agg_fn = {
    'median': np.median,
    'mean': np.mean,
    'max': np.max,
    'min': np.min
}


def ilisi(
    datasets: List[Dataset],
    agg: Literal['median', 'mean'] = 'median',
    perplexity: int = 30
):
    latents = np.concatenate([d.latent for d in datasets])
    obs = pd.concat([d.obs for d in datasets])
    ilisi_dist = compute_lisi(latents, obs, ['source'], perplexity)
    ilisi_agg = agg_fn[agg](ilisi_dist)
    return ilisi_agg


def clisi(
    datasets: List[Dataset],
    agg: Literal['median', 'mean'] = 'median',
    perplexity: int = 30
):
    latents = np.concatenate([d.latent for d in datasets])
    obs = pd.concat([d.obs for d in datasets])
    clisi_dist = compute_lisi(latents, obs, ['cell_type'], perplexity)
    clisi_agg = agg_fn[agg](clisi_dist)
    return clisi_agg


def f1_lisi(
    datasets: List[Dataset],
    perplexity: int = 30
):
    latents = np.concatenate([d.latent for d in datasets])
    obs = pd.concat([d.obs for d in datasets])
    lisi_dist = compute_lisi(latents, obs, ['source', 'cell_type'], perplexity)
    ilisi_dist = lisi_dist[:, 0]
    clisi_dist = lisi_dist[:, 1]
    ilisi_norm = (agg_fn['median'](ilisi_dist) - agg_fn['min'](ilisi_dist)) / \
                 (agg_fn['max'](ilisi_dist) - agg_fn['min'](ilisi_dist))
    clisi_norm = (agg_fn['median'](clisi_dist) - agg_fn['min'](clisi_dist)) / \
                 (agg_fn['max'](clisi_dist) - agg_fn['min'](clisi_dist))
    f1_lisi_norm = 2 * (1 - clisi_norm) * ilisi_norm / (1 - clisi_norm + ilisi_norm)
    return f1_lisi_norm
