import os
import warnings
from typing import Optional

from flask import Flask
from flask_cors import CORS

from polyphony import Polyphony
from polyphony.data import load_pancreas, load_pbmc
from polyphony.router.routes import add_routes
from polyphony.router.utils import create_project_folders, NpEncoder, SERVER_STATIC_DIR
from polyphony.tool._projection import umap_transform


def create_app(args):
    if args.warnings is None or not args.warnings:
        warnings.filterwarnings("ignore")

    app = Flask(
        __name__,
        static_url_path='/files/',
        static_folder=SERVER_STATIC_DIR,
        template_folder='../../apidocs'
    )

    app.json_encoder = NpEncoder
    app.folders = create_project_folders(args.experiment, SERVER_STATIC_DIR)

    ref_dataset, qry_dataset = load_datasets(args.problem, args.iter)
    app.pp = Polyphony(args.experiment, ref_dataset, qry_dataset)

    if args.iter is None:
        # step-0: set up the anndata in scvi-tools
        app.pp.setup_anndata()

        # step-1: build the reference model and initialize the query model
        app.pp.init_reference_step()

        # step-2: calculate the 2-D UMAP projections according to the cells' latent representations
        umap_transform(ref_dataset, qry_dataset)

        # step-3: recommend anchor candidates
        app.pp.recommend_anchors()

    # save anndata in .zarr format
    ref_dataset.adata.write_zarr(os.path.join(app.folders['zarr'], 'reference.zarr'))
    qry_dataset.adata.write_zarr(os.path.join(app.folders['zarr'], 'query.zarr'))

    CORS(app, resources={r"/api/*": {"origins": "*"}, r"/files/*": {"origins": "*"}})
    add_routes(app, SERVER_STATIC_DIR)

    return app


def load_datasets(problem_id: str, iter: Optional[int]):
    if iter is None:
        if problem_id == 'pancreas':
            ref_dataset, qry_dataset = load_pancreas()
        elif problem_id == 'pbmc':
            ref_dataset, qry_dataset = load_pbmc()
        elif problem_id == 'pbmc_exp':
            ref_dataset, qry_dataset = load_pbmc(remove_cell_type='Plasmacytoid dendritic cells')
        else:
            raise ValueError("Unknown problem.")
    else:
        raise NotImplementedError
    return ref_dataset, qry_dataset
