from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse
import os


app = Flask(__name__)
api = Api(app)
port = int(os.environ.get("PORT", 5000))

parser = reqparse.RequestParser()
parser.add_argument('number', type=float, required=True)

class HelloWorld(Resource):
    def get(self):
        return {'hello':'world'}

class Square(Resource):
    def post(self):
        args = parser.parse_args()
        number = args['number']
        res = number * number
        return {'square': res}, 200


api.add_resource(HelloWorld, '/hello')
api.add_resource(Square, '/square')

if __name__ == '__main__':
	app.run(debug=True, host='0.0.0.0',port=port)
