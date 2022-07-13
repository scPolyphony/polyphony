from flask import current_app
from flask_restful import reqparse, Resource

from polyphony import Polyphony


class AnchorResource(Resource):

    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('anchor', type=dict, location='json')
        self.parser.add_argument('anchor_id', location='json')
        self.parser.add_argument('operation', location='json')
        self.pp: Polyphony = current_app.pp

    def get(self):
        anchors = {"unjustified": [], "confirmed": [], "user_selection": []}
        for anchor in self.pp.anchors:
            if anchor.confirmed:
                anchors["confirmed"].append(anchor.to_dict())
            elif anchor.create_by == "model":
                anchors["unjustified"].append(anchor.to_dict())
            else:
                anchors["user_selection"].append(anchor.to_dict())
        return anchors

    def put(self):
        args = self.parser.parse_args()
        print(args)
        anchor = args.get('anchor', None)
        anchor_id = args.get('anchor_id', None)
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
