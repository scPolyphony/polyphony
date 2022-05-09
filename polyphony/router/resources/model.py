import os

from flask import current_app
from flask_restful import reqparse, Resource

from polyphony import Polyphony
from polyphony.tool._projection import umap_transform


class ModelResource(Resource):

    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('operation', location='json')
        self.pp: Polyphony = current_app.pp

    def put(self):
        args = self.parser.parse_args()
        operation = args.get('operation', None)
        if operation == 'update':
            zarr_folder = current_app.folders['zarr']

            self.pp.setup_anndata_anchors(self.pp.confirmed_anchors)
            self.pp.model_update_step()
            umap_transform(self.pp.ref, self.pp.qry)
            current_app.pp.recommend_anchors()

            self.pp.ref.adata.write_zarr(os.path.join(zarr_folder, 'reference.zarr'))
            self.pp.qry.adata.write_zarr(os.path.join(zarr_folder, 'query.zarr'))
            return 'success'
        else:
            raise ValueError("Invalid operation.")

    def delete(self):
        args = self.parser.parse_args()
        self.pp.delete_anchor(args.get('id'))
