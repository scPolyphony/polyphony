from flask import current_app, send_from_directory
from flask_restful import Api

import polyphony.router.resources as res

API = '/api/'


def add_routes(app, static_dir):
    api = Api(app)

    @app.route('/files/<path:path>')
    def get_file(path):
        return send_from_directory(static_dir, filename=path, as_attachment=True)

    api.add_resource(res.AnchorResource, API + 'anchor')

    @app.route('/api/model_update')
    def model_update():
        current_app.pm.update_round()
        return 'success'
