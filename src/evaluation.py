# src/evaluation.py
import os
import matplotlib.pyplot as plt
from client import run_client
from centralized_training_all_features import train_and_evaluate_model as train_all_features
from centralized_feature_selection_training import centralized_feature_selection, train_and_evaluate_model as train_selected_features
from data_loader import load_and_split_data
import numpy as np

def evaluate_federated_vs_centralized(num_clients=5, num_features=5):
    # Load data
    client_data, (X_test, y_test), feature_names = load_and_split_data(num_clients=num_clients)

    # Step 1: Federated Feature Selection
    print("Running Federated Feature Selection...")
    for client_id in range(num_clients):
        run_client(client_id)  # Each client runs individually to simulate federated setup
    
    # Assuming global ranking is aggregated and provided by clients
    # Placeholder for federated model metrics (assuming global ranking is available)
    federated_results = []
    for client_id in range(num_clients):
        results_path = f"results/federated_results_client_{client_id}.txt"
        with open(results_path, "r") as f:
            metrics = f.readlines()[-1].strip().split(', ')
            federated_results.append([float(metric.split(': ')[1]) for metric in metrics])
    federated_results = np.mean(federated_results, axis=0)
    federated_accuracy, federated_f1, federated_precision, federated_recall, federated_auc = federated_results

    # Step 2: Centralized Feature Selection
    print("Running Centralized Feature Selection...")
    X_train = np.vstack([client[0] for client in client_data])
    y_train = np.hstack([client[1] for client in client_data])

    # Perform centralized feature selection
    selected_features = centralized_feature_selection(X_train, y_train, num_features=num_features)
    X_train_selected = X_train[:, selected_features]
    X_test_selected = X_test[:, selected_features]

    # Train and evaluate the model with selected features
    cent_accuracy, cent_f1, cent_precision, cent_recall, cent_auc = train_selected_features(X_train_selected, y_train, X_test_selected, y_test)
    print("\nCentralized Feature Selection Model:")
    print(f"Accuracy: {cent_accuracy:.4f}, F1 Score: {cent_f1:.4f}, Precision: {cent_precision:.4f}, Recall: {cent_recall:.4f}, AUC: {cent_auc:.4f}")

    # Step 3: Centralized Training with All Features
    print("Running Centralized Training with All Features...")
    all_accuracy, all_f1, all_precision, all_recall, all_auc = train_all_features(X_train, y_train, X_test, y_test)
    print("\nCentralized Training with All Features:")
    print(f"Accuracy: {all_accuracy:.4f}, F1 Score: {all_f1:.4f}, Precision: {all_precision:.4f}, Recall: {all_recall:.4f}, AUC: {all_auc:.4f}")

    # Step 4: Save Results
    results_path = "results/federated_vs_centralized_comparison.txt"
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, "a") as f:
        f.write("Federated vs Centralized Comparison Metrics\n")
        f.write("Federated Feature Selection:\n")
        f.write(f"Accuracy: {federated_accuracy:.4f}, F1 Score: {federated_f1:.4f}, Precision: {federated_precision:.4f}, Recall: {federated_recall:.4f}, AUC: {federated_auc:.4f}\n\n")
        f.write("Centralized Feature Selection:\n")
        f.write(f"Accuracy: {cent_accuracy:.4f}, F1 Score: {cent_f1:.4f}, Precision: {cent_precision:.4f}, Recall: {cent_recall:.4f}, AUC: {cent_auc:.4f}\n\n")
        f.write("Centralized Training with All Features:\n")
        f.write(f"Accuracy: {all_accuracy:.4f}, F1 Score: {all_f1:.4f}, Precision: {all_precision:.4f}, Recall: {all_recall:.4f}, AUC: {all_auc:.4f}\n\n")

    # Step 5: Generate and Save Plot
    plot_path = "results/plots/performance_comparison.png"
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'AUC']
    federated_values = [federated_accuracy, federated_f1, federated_precision, federated_recall, federated_auc]
    cent_values = [cent_accuracy, cent_f1, cent_precision, cent_recall, cent_auc]
    all_values = [all_accuracy, all_f1, all_precision, all_recall, all_auc]

    x = np.arange(len(metrics))
    width = 0.25

    plt.figure()
    plt.bar(x - width, federated_values, width, label='Federated')
    plt.bar(x, cent_values, width, label='Centralized (Selected Features)')
    plt.bar(x + width, all_values, width, label='Centralized (All Features)')

    plt.xlabel('Evaluation Metric')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

if __name__ == "__main__":
    evaluate_federated_vs_centralized()