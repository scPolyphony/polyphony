from typing import List, Optional, Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from scarches.models import SCVI
from scvi.model._utils import _init_library_size

from polyphony.anchor import Anchor
from polyphony.data import RefAnnDataManager, QryAnnDataManager
from polyphony.models import ActiveVAE

UPDATE_KEY = '_scvi_cell_update'
REP_KEY = '_scvi_desired_rep'


class ActiveSCVI(SCVI):
    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        **model_kwargs
    ):
        super(ActiveSCVI, self).__init__(
            adata,
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
            dispersion,
            gene_likelihood,
            latent_distribution,
            **model_kwargs
        )

        n_cats_per_cov = (
            self.scvi_setup_dict_["extra_categoricals"]["n_cats_per_key"]
            if "extra_categoricals" in self.scvi_setup_dict_
            else None
        )
        n_batch = self.summary_stats["n_batch"]
        library_log_means, library_log_vars = _init_library_size(adata, n_batch)

        self.module = ActiveVAE(
            n_input=self.summary_stats["n_vars"],
            n_batch=n_batch,
            n_labels=self.summary_stats["n_labels"],
            n_continuous_cov=self.summary_stats["n_continuous_covs"],
            n_cats_per_cov=n_cats_per_cov,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            **model_kwargs,
        )

    @staticmethod
    def setup_anndata(
        adata: AnnData,
        n_latent: int = 10,
        **kwargs
    ):
        SCVI.setup_anndata(
            adata,
            **kwargs
        )

        # Register `update_key`, which indicates whether the cell's latent vector has
        # already be given
        data_registry = adata.uns['_scvi']['data_registry']

        # Fill the (whether to) update field with False
        if UPDATE_KEY not in adata.obs.keys():
            adata.obs[UPDATE_KEY] = False
        data_registry['cell_update'] = {'attr_name': 'obs', 'attr_key': UPDATE_KEY}

        # Register `update_rep_key` in `adata.obsm`, which includes cells' (desired) latent vectors
        if REP_KEY not in adata.obsm.keys():
            adata.obsm[REP_KEY] = np.zeros([len(adata), n_latent])

        # TODO: registering obsm keys is not supported by the `_setup_anndata` function in
        #  scvi-tool == 14.6. So it is manually done.
        data_registry['desired_rep'] = {'attr_name': 'obsm', 'attr_key': REP_KEY}

    @staticmethod
    def setup_anchor_rep(
        ref_dataset: RefAnnDataManager,
        qry_dataset: QryAnnDataManager,
        confirmed_anchors: List[Anchor] = None
    ):
        if confirmed_anchors is None:
            return
        assign_mask = np.zeros(qry_dataset.anchor_prob.shape)
        cell_locs = []
        for anchor in confirmed_anchors:
            cells = [info['cell_id'] for info in anchor.cells]
            cell_loc = qry_dataset.adata.obs.index.get_indexer_for(cells)
            cell_locs += cell_loc.tolist()
            assign_mask[cell_loc, anchor.reference_id] = 1
            qry_dataset.adata.obs[UPDATE_KEY].loc[cells] = True
        anchor_mat = assign_mask * qry_dataset.anchor_prob
        phi = pd.get_dummies(qry_dataset.batch).to_numpy().T
        phi_moe = np.vstack((np.repeat(1, phi.shape[1]), phi))

        compression_terms = ref_dataset.adata.uns['_polyphony_params']['compression_terms']
        corr = symphony_correct(qry_dataset.latent, anchor_mat.T, compression_terms, phi_moe)
        qry_dataset.adata.obsm[REP_KEY][cell_locs] = corr[cell_locs]


def symphony_correct(latent, R, compression_terms, phi_moe):
    lamb = 1
    N = compression_terms['N']
    C = compression_terms['C']
    updated_latent = latent.copy()
    for i in range(R.shape[0]):
        E = np.dot(np.dot(phi_moe, np.diag(R[i, :])), phi_moe.T)
        E[0, 0] += N[i]
        F = np.dot(np.dot(phi_moe, np.diag(R[i, :])), latent)
        F[0, :] += C[i, :]
        B = np.dot(np.linalg.inv(E + np.diag([lamb] * E.shape[0])), F)
        B[0, :] = 0
        updated_latent -= np.dot(np.dot(B.T, phi_moe), np.diag(R[i, :])).T
    return updated_latent
