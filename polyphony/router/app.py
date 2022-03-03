from flask import Flask
from flask_cors import CORS

from polyphony.router.routes import add_routes
from polyphony.router.polyphony_manager import PolyphonyManager
from polyphony.router.utils import SERVER_STATIC_DIR


def create_app(problem_id):
    pm = PolyphonyManager(problem_id, SERVER_STATIC_DIR)
    pm.init_round()

    app = Flask(
        __name__,
        static_url_path='/files/',
        static_folder=SERVER_STATIC_DIR,
        template_folder='../../apidocs'
    )

    app.pm = pm

    CORS(app, resources={r"/api/*": {"origins": "*"}, r"/files/*": {"origins": "*"}})
    add_routes(app, SERVER_STATIC_DIR)

    return app
