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
        # args = current_app.args
        # load_exist = args.load_exist and args.iter is not None
        # current_app.pm.update_round(load_exist=load_exist)
        current_app.pm.update_round(save=current_app.args.save, eval=current_app.args.eval)
        return 'success'
