from flask import send_from_directory, current_app, jsonify, request
from flask_restful import Api

API = '/api/'


def add_routes(app, static_dir):
    api = Api(app)

    @app.route('/files/<path:path>')
    def get_file(path):
        return send_from_directory(static_dir, filename=path, as_attachment=True)

    @app.route(API + 'anchor', methods=['GET'])
    def get_anchor():
        return jsonify(current_app.pm.anchor_to_json())

    @app.route(API + 'confirmed_anchor', methods=['POST'])
    def post_confirmed_anchor():
        request_params = request.get_json()
        current_app.pm.update_round(request_params)
        return "succeed"
