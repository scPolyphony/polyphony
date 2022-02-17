import numpy as np

from typing import Optional

from anndata import AnnData
from scarches.models import SCVI
from scvi.model._utils import _init_library_size
from scvi._compat import Literal

from polyphony.models import ActiveVAE


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
        update_key: Optional[str] = None,
        update_latent_key: Optional[str] = None,
        n_latent: int = 10,
        **kwargs
    ):
        SCVI.setup_anndata(
            adata,
            **kwargs
        )

        # Register `update_key`, which indicates whether the cell's latent vector has
        # already be given
        update_key = 'cell_update' if update_key is None else update_key

        # Fill the (whether to) update field with False
        if update_key not in adata.obs.keys():
            adata.obs[update_key] = False
        adata.uns['_scvi']['data_registry']['cell_update'] = {'attr_name': 'obs',
                                                              'attr_key': update_key}

        # Register `update_rep_key` in `adata.obsm`, which includes cells' (desired) latent vectors
        update_rep_key = 'desired_rep' if update_latent_key is None else update_latent_key
        if update_rep_key not in adata.obsm.keys():
            adata.obsm[update_rep_key] = np.zeros([len(adata), n_latent])

        # TODO: registering obsm keys is not supported by the `_setup_anndata` function in
        #  scvi-tool == 14.6. So it is manually done.
        adata.uns['_scvi']['data_registry']['desired_rep'] = {'attr_name': 'obsm',
                                                              'attr_key': update_rep_key}
