import os
import warnings
from typing import Optional

from flask import Flask, send_from_directory
from flask_cors import CORS

from polyphony import Polyphony
from polyphony.data import load_pancreas, load_pbmc
from polyphony.router.routes import add_routes
from polyphony.tool._projection import umap_transform
from polyphony.utils._dir_manager import DirManager
from polyphony.utils._json_encoder import NpEncoder

SERVER_STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')


def create_app(args):
    if args.warnings is None or not args.warnings:
        warnings.filterwarnings("ignore")

    app = Flask(
        __name__,
        static_url_path='/files/',
        static_folder=SERVER_STATIC_DIR,
        template_folder='../../apidocs'
    )

    @app.route('/files/<path:path>')
    def get_file(path):
        return send_from_directory(SERVER_STATIC_DIR, filename=path, as_attachment=True)

    app.json_encoder = NpEncoder
    app.folders = create_project_folders(args.experiment, SERVER_STATIC_DIR)

    ref_dataset, qry_dataset = load_datasets(args.problem, args.experiment, args.iter)
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
    qry_dataset.anchor = None
    ref_dataset.adata.write_zarr(os.path.join(app.folders['zarr'], 'reference.zarr'))
    qry_dataset.adata.write_zarr(os.path.join(app.folders['zarr'], 'query.zarr'))

    CORS(app, resources={r"/api/*": {"origins": "*"}, r"/files/*": {"origins": "*"}})
    add_routes(app)

    return app


def load_datasets(problem_id: str, experiment: Optional[str] = None, iter: Optional[int] = None):
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
        dir_manager = DirManager(instance_id=experiment)
        ref_dataset = dir_manager.load_data(data_type='ref', model_iter=iter)
        qry_dataset = dir_manager.load_data(data_type='qry', model_iter=iter)
    return ref_dataset, qry_dataset


def create_project_folders(problem_id, root_dir=SERVER_STATIC_DIR, extensions=None):
    extensions = ['zarr', 'json', 'csv'] if extensions is None else extensions
    folders = {}
    for ex in extensions:
        fpath = os.path.join(root_dir, ex, problem_id)
        folders[ex] = fpath
        os.makedirs(fpath, exist_ok=True)
    return folders
