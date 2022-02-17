import scanpy as sc
import matplotlib.pyplot as plt


def display_umap(matrix, umap_embedder, latent_key='latent', fit=False, color=None):
    if fit:
        projected = umap_embedder.fit_transform(matrix.obsm[latent_key])
    else:
        projected = umap_embedder.transform(matrix.obsm[latent_key])
    matrix.obsm['X_umap'] = projected
    plt.figure()
    sc.pl.umap(matrix, color=color, frameon=False, wspace=0.6)
