# src/data_loader.py
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import os

def load_and_split_data(num_clients=5):
    # Load the dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Split the training data among clients
    client_data = []
    client_data_size = len(X_train) // num_clients
    for i in range(num_clients):
        start_idx = i * client_data_size
        end_idx = (i + 1) * client_data_size if i != num_clients - 1 else len(X_train)
        X_client, y_client = X_train[start_idx:end_idx], y_train[start_idx:end_idx]
        client_data.append((X_client, y_client))

    # Save data splits for reproducibility
    results_path = "results/data_splits/"
    os.makedirs(results_path, exist_ok=True)
    np.save(os.path.join(results_path, "X_test.npy"), X_test)
    np.save(os.path.join(results_path, "y_test.npy"), y_test)
    for i, (X_client, y_client) in enumerate(client_data):
        np.save(os.path.join(results_path, f"X_client_{i}.npy"), X_client)
        np.save(os.path.join(results_path, f"y_client_{i}.npy"), y_client)

    return client_data, (X_test, y_test), data.feature_names

if __name__ == "__main__":
    clients, test_data, feature_names = load_and_split_data()
    print(f"Loaded data for {len(clients)} clients.")
