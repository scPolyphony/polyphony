import scanpy as sc
import matplotlib.pyplot as plt


def display_umap(matrix, color=None):
    plt.figure()
    sc.pl.umap(matrix, color=color, frameon=False, wspace=0.6)
