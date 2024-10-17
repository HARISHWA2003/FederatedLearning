# src/client.py
import requests
import numpy as np
from feature_selection import local_feature_ranking
from data_loader import load_and_split_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import os

SERVER_URL = "http://<server_ip>:5000"  # Replace <server_ip> with server's IP address

def run_client(client_id):
    # Load client-specific data
    client_data, _, _ = load_and_split_data(num_clients=5)
    X, y = client_data[client_id]

    # Perform local feature ranking
    ranking, scores = local_feature_ranking(X, y)
    ranking = scores.tolist()

    # Send rankings to server
    response = requests.post(f"{SERVER_URL}/send_rankings", json={"ranking": ranking})
    if response.status_code == 200:
        print(f"Client {client_id}: Rankings sent successfully.")

    # Fetch aggregated ranking from server
    response = requests.get(f"{SERVER_URL}/aggregate_rankings")
    if response.status_code == 200:
        global_ranking = response.json()['global_ranking']
        print(f"Client {client_id}: Received global ranking: {global_ranking}")

        # Train model using global ranking
        selected_features = global_ranking[:5]  # Select top 5 features for demonstration
        X_selected = X[:, selected_features]
        model = RandomForestClassifier(random_state=42)
        model.fit(X_selected, y)
        predictions = model.predict(X_selected)

        # Calculate metrics
        accuracy = accuracy_score(y, predictions)
        f1 = f1_score(y, predictions)
        precision = precision_score(y, predictions)
        recall = recall_score(y, predictions)
        auc = roc_auc_score(y, model.predict_proba(X_selected)[:, 1])

        # Save metrics
        results_path = f"results/federated_results_client_{client_id}.txt"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, "a") as f:
            f.write(f"Client {client_id}: Global Feature Ranking: {global_ranking}\n")
            f.write(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}\n\n")
    else:
        print(f"Client {client_id}: Failed to receive global ranking.")

if __name__ == '__main__':
    run_client(client_id=0)  # Set unique ID for each client