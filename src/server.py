# src/server.py
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

global_feature_rankings = []

@app.route('/send_rankings', methods=['POST'])
def receive_rankings():
    global global_feature_rankings
    client_ranking = request.json['ranking']
    global_feature_rankings.append(client_ranking)
    return jsonify({"message": "Rankings received"}), 200

@app.route('/aggregate_rankings', methods=['GET'])
def aggregate_rankings():
    if not global_feature_rankings:
        return jsonify({"message": "No rankings to aggregate"}), 400
    aggregated_scores = np.mean(global_feature_rankings, axis=0)
    global_ranking = np.argsort(-aggregated_scores).tolist()
    global_feature_rankings = []  # Reset for the next round
    return jsonify({"global_ranking": global_ranking}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)