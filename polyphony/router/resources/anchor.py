from flask import current_app
from flask_restful import reqparse, Resource

from polyphony.router.polyphony_manager import PolyphonyManager


class AnchorResource(Resource):

    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('anchor')
        self.parser.add_argument('anchor_id')
        self.parser.add_argument('operation')
        self.pm: PolyphonyManager = current_app.pm

    def get(self):
        return self.pm.get_anchor()

    def put(self):
        args = self.parser.parse_args()
        self.pm.put_anchor(args)

    def delete(self):
        args = self.parser.parse_args()
        self.pm.delete_anchor(args.get('anchor_id'))
