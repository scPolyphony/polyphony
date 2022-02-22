import os

from flask import Flask, send_from_directory
from flask_cors import CORS

from polyphony import Polyphony
from polyphony.dataset import load_pancreas
from polyphony.router.routes import add_routes


SERVER_STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')


def create_app():
    problem_id = 'pancreas'

    if problem_id == 'pancreas':
        ref_dataset, query_dataset = load_pancreas()
    else:
        raise NotImplemented

    pp = Polyphony(ref_dataset, query_dataset)
    pp.setup_data()
    pp.init_reference_step()
    pp.umap_transform()

    pp.ref_dataset.save_adata(os.path.join(SERVER_STATIC_DIR, 'zarr',
                                           problem_id + 'reference.zarr'))
    pp.query_dataset.save_adata(os.path.join(SERVER_STATIC_DIR, 'zarr',
                                             problem_id + 'query.zarr'))

    app = Flask(
        __name__,
        static_url_path='/files/',
        static_folder=SERVER_STATIC_DIR,
        template_folder='../../apidocs'
    )

    @app.route('/files/<path:path>')
    def get_file(path):
        return send_from_directory(SERVER_STATIC_DIR, filename=path, as_attachment=True)

    app.pp = pp

    CORS(app, resources={r"/api/*": {"origins": "*"}, r"/files/*": {"origins": "*"}})
    add_routes(app)

    return app


def init_pp():
    pass

