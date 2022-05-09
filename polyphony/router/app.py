import os
import warnings

from flask import Flask, send_from_directory
from flask_cors import CORS

from polyphony import Polyphony
from polyphony.data import load_pancreas, load_pbmc
from polyphony.router.routes import add_routes
from polyphony.tool._projection import umap_transform
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
    app.args = args
    app.folders = create_project_folders(args.experiment, SERVER_STATIC_DIR)

    if args.iter is not None and args.load_exist:
        pp = Polyphony.load_snapshot(args.experiment, args.iter)
    else:
        ref_dataset, qry_dataset = load_datasets(args.problem)
        pp = Polyphony(args.experiment, ref_dataset, qry_dataset)

    if args.iter is None:
        # step-0: set up the anndata in scvi-tools
        pp.setup_anndata()

        # step-1: build the reference model and initialize the query model
        pp.init_reference_step()

        # step-2: calculate the 2-D UMAP projections according to the cells' latent representations
        umap_transform(pp.ref, pp.qry)

        # step-3: recommend anchor candidates
        pp.recommend_anchors()

        if args.save:
            pp.save_snapshot()

    # save anndata in .zarr format
    pp.qry.anchor = None
    pp.ref.adata.write_zarr(os.path.join(app.folders['zarr'], 'reference.zarr'))
    pp.qry.adata.write_zarr(os.path.join(app.folders['zarr'], 'query.zarr'))

    app.pp = pp

    CORS(app, resources={r"/api/*": {"origins": "*"}, r"/files/*": {"origins": "*"}})
    add_routes(app)

    return app


def load_datasets(problem_id: str):
    if problem_id == 'pancreas':
        ref_dataset, qry_dataset = load_pancreas()
    elif problem_id == 'pbmc':
        ref_dataset, qry_dataset = load_pbmc()
    elif problem_id == 'pbmc_exp':
        ref_dataset, qry_dataset = load_pbmc(remove_cell_type='Plasmacytoid dendritic cells')
    else:
        raise ValueError("Unknown problem.")
    return ref_dataset, qry_dataset


def create_project_folders(problem_id, root_dir=SERVER_STATIC_DIR, extensions=None):
    extensions = ['zarr', 'json', 'csv'] if extensions is None else extensions
    folders = {}
    for ex in extensions:
        fpath = os.path.join(root_dir, ex, problem_id)
        folders[ex] = fpath
        os.makedirs(fpath, exist_ok=True)
    return folders
