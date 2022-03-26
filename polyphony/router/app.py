from flask import Flask
from flask_cors import CORS

from polyphony.router.routes import add_routes
from polyphony.router.polyphony_manager import PolyphonyManager
from polyphony.router.utils import NpEncoder, SERVER_STATIC_DIR


def create_app(args):
    print(args)
    pm = PolyphonyManager(args.problem, args.experiment, args.iter, static_folder=SERVER_STATIC_DIR)
    if args.iter is None:
        pm.init_round(load_exist=args.load_exist, save=args.save)
    else:
        pm.save_ann()

    app = Flask(
        __name__,
        static_url_path='/files/',
        static_folder=SERVER_STATIC_DIR,
        template_folder='../../apidocs'
    )

    app.json_encoder = NpEncoder
    app.pm = pm
    app.args = args

    CORS(app, resources={r"/api/*": {"origins": "*"}, r"/files/*": {"origins": "*"}})
    add_routes(app, SERVER_STATIC_DIR)

    return app
