import simtool.dataloader as d
from simtool.serving import ModelServer

from flask import Flask, jsonify, make_response, request

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        path = request.json['path']
        k = request.json['k']
        prediction = server.query_topk(path,k)
        return make_response(jsonify({'neighbors': prediction}))
    except KeyError:
        raise RuntimeError('Key cannot be be found in JSON payload.')


if __name__ == '__main__':
    server = ModelServer()
    app.run(host='0.0.0.0', port=5000)
