# How to Perform the Experiment and Check the Results

### Step 1: Setting Up the Server
- The server is responsible for aggregating feature rankings from all clients. Start the server by running:
```sh
python src/server.py
```
- This will initialize the server, which listens for incoming feature rankings from clients and aggregates them to determine the final global feature importance.

### Step 2: Setting Up Clients
- Update the `SERVER_URL` in `client.py` to match the IP address of the server (e.g., `http://<server_ip>:5000`).
- Each client computes feature rankings locally and sends them to the server.
- Run the client script by specifying the client ID:
```sh
python src/client.py
```
- You can run multiple clients on different machines or on the same machine with different `client_id`s to simulate a federated environment.

### Step 3: Aggregating Feature Rankings
- Once all clients have sent their feature rankings to the server, the server aggregates these rankings and determines the global feature subset.
- Clients then use the global feature ranking to train a local model using the top-ranked features.

### Step 4: Centralized Training with Feature Selection
- To compare the federated approach with a centralized approach, perform centralized feature selection using the full dataset.
- Run the following script to perform centralized feature selection and training:
```sh
python src/centralized_feature_selection_training.py
```
- This script uses mutual information to select the most important features and then trains a model using only those features.

### Step 5: Centralized Training with All Features
- To provide a baseline for comparison, run centralized training using all available features:
```sh
python src/centralized_training_all_features.py
```
- This script trains a model without any feature selection, using all features in the dataset.

### Step 6: Evaluating the Results
- Use the `evaluation.py` script to generate comparisons between the federated and centralized scenarios.
- This script will output key metrics such as **accuracy**, **F1 score**, **precision**, **recall**, and **AUC** to help understand the performance differences between the approaches.
```sh
python src/evaluation.py
```
- The evaluation results, along with plots, will be saved in the `results/` directory for further analysis.

### Checking the Results
- The **results** are saved in the `results/` directory:
  - **Federated Results**: Metrics for each client are stored in files like `results/federated_results_client_<client_id>.txt`.
  - **Centralized Results**: Metrics for centralized training with feature selection and with all features are stored in `results/centralized_results.txt`.
  - **Comparison Metrics**: The aggregated comparison between federated and centralized methods is saved in `results/federated_vs_centralized_comparison.txt`.
  - **Plots**: A visual comparison of model performance is saved as `results/plots/performance_comparison.png`.

### Results and Analysis
- The experiment compares three scenarios:
  1. **Federated Feature Selection**: Collaborative feature selection across clients without sharing raw data.
  2. **Centralized Feature Selection**: Feature selection done on the entire dataset in a centralized manner.
  3. **Centralized Training with All Features**: Baseline model trained with all features.
- **Metrics** such as accuracy, precision, recall, F1 score, and AUC are used to evaluate and compare model performance.
- The comparison metrics and plots help understand the effectiveness of federated learning in identifying key features while preserving data privacy.

