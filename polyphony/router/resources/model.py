import os

from flask import current_app
from flask_restful import Resource

from polyphony import Polyphony
from polyphony.tool._projection import umap_transform


class ModelResource(Resource):

    def __init__(self):
        self.pp: Polyphony = current_app.pp
        self.args = current_app.args

    def get(self):
        zarr_folder = current_app.folders['zarr']

        self.pp.setup_anndata_anchors(self.pp.confirmed_anchors)
        self.pp.model_update_step()
        umap_transform(self.pp.ref, self.pp.qry)
        current_app.pp.recommend_anchors()

        if self.args.save:
            self.pp.save_snapshot()

        self.pp.ref.adata.write_zarr(os.path.join(zarr_folder, 'reference.zarr'))
        self.pp.qry.adata.write_zarr(os.path.join(zarr_folder, 'query.zarr'))
        return 'success'
