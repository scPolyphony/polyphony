from flask import send_from_directory
from flask_restful import Api

import polyphony.router.resources as res

API = '/api/'


def add_routes(app):
    api = Api(app)

    # anchor apis
    api.add_resource(res.AnchorResource, API + 'anchor')

    # model apis
    api.add_resource(res.ModelResource, API + 'model')
