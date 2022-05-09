from flask import current_app
from flask_restful import reqparse, Resource

from polyphony import Polyphony


class AnchorResource(Resource):

    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('anchor', type=dict, location='json')
        self.parser.add_argument('id', location='json')
        self.parser.add_argument('operation', location='json')
        self.pp: Polyphony = current_app.pp

    def get(self):
        return [anchor.to_dict() for anchor in self.pp.anchors]

    def put(self):
        args = self.parser.parse_args()
        anchor = args.get('anchor', None)
        anchor_id = args.get('id', None)
        operation = args.get('operation', None)  # be either 'add', 'refine', and 'confirm'
        if operation == 'add':
            self.pp.register_anchor(anchor)
        elif operation == 'refine':
            self.pp.refine_anchor(anchor)
        elif operation == 'confirm':
            self.pp.confirm_anchor(anchor_id)
        else:
            raise ValueError("Invalid operation.")

    def delete(self):
        args = self.parser.parse_args()
        self.pp.delete_anchor(args.get('id'))
