from flask import Flask
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)

def preds(year, month):
    # Use the ML model to make predictions based on the year and month add in future
    if year == 2018 and month == 1:
        return 100
    else:
        return 0

class predict(Resource):
    def post(self):
        # the api will take in a json file with the year and month
        # and return the prediction
        parse = reqparse.RequestParser()
        parse.add_argument('year', type=int, required=True)
        parse.add_argument('month', type=int, required=True)
        args = parse.parse_args()
        year = args['year']
        month = args['month']
        prediction = preds(year, month)
        return prediction, 200
api = Api(app)
api.add_resource(predict, '/predict')

if __name__ == '_main_':
    app.run(port=5000,debug=True)