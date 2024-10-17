# src/centralized_feature_selection_training.py
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from data_loader import load_and_split_data
import numpy as np
import os

def centralized_feature_selection(X_train, y_train, num_features=5):
    # Centralized feature ranking using mutual information
    scores = mutual_info_classif(X_train, y_train)
    ranking = np.argsort(-scores)[:num_features]
    return ranking

def train_and_evaluate_model(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # Evaluation Metrics
    accuracy = accuracy_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    # Save results
    results_path = "results/centralized_results.txt"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "a") as f:
        f.write(f"Centralized Training with Feature Selection\n")
        f.write(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}\n\n")

    return accuracy, f1, precision, recall, auc

if __name__ == "__main__":
    # Load data
    client_data, (X_test, y_test), feature_names = load_and_split_data(num_clients=5)
    X_train = np.vstack([client[0] for client in client_data])
    y_train = np.hstack([client[1] for client in client_data])

    # Perform centralized feature selection
    selected_features = centralized_feature_selection(X_train, y_train, num_features=5)
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    # Train and evaluate the model
    train_and_evaluate_model(X_train_selected, y_train, X_test_selected, y_test)