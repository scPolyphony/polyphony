from flask import send_from_directory
from flask_restful import Api

API = '/api/'


def add_routes(app, static_dir):
    api = Api(app)

    @app.route('/files/<path:path>')
    def get_file(path):
        return send_from_directory(static_dir, filename=path, as_attachment=True)
