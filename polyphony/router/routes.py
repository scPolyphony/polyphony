import os

from flask import current_app, send_from_directory
from flask_restful import Api

import polyphony.router.resources as res
from polyphony import Polyphony
from polyphony.tool._projection import umap_transform

API = '/api/'


def add_routes(app, static_dir):
    api = Api(app)

    @app.route('/files/<path:path>')
    def get_file(path):
        return send_from_directory(static_dir, filename=path, as_attachment=True)

    api.add_resource(res.AnchorResource, API + 'anchor')

    @app.route('/api/model_update')
    def model_update():
        pp: Polyphony = current_app.pp
        zarr_folder = current_app.folders['zarr']

        pp.setup_anndata_anchors(pp.confirmed_anchors)
        pp.model_update_step()
        umap_transform(pp.ref, pp.qry)
        app.pp.recommend_anchors()

        pp.ref.adata.write_zarr(os.path.join(zarr_folder, 'reference.zarr'))
        pp.qry.adata.write_zarr(os.path.join(zarr_folder, 'query.zarr'))
        return 'success'
