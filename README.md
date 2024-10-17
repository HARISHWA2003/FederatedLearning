# Federated Learning with Federated Feature Selection and Centralized Comparisons

## Project Overview
This project implements a federated learning system with federated feature selection and compares it with traditional centralized methods. The aim is to evaluate the effectiveness of federated feature selection in identifying important features in a privacy-preserving manner while comparing its performance with centralized feature selection and centralized training.

### Key Components
1. **Federated Feature Selection**: Clients compute feature importance rankings locally and send them to a central server, which aggregates them to determine the global feature subset.
2. **Centralized Feature Selection**: A centralized feature selection method is applied using mutual information.
3. **Centralized Training**: Models are trained using all features to provide a baseline comparison.

### Directory Structure
```
project_root/
├── src/
│   ├── client.py                           # Federated learning client code
│   ├── server.py                           # Federated learning server code
│   ├── centralized_training_all_features.py # Centralized training with all features
│   ├── centralized_feature_selection_training.py # Centralized training with feature selection
│   ├── data_loader.py                      # Script for loading and splitting data
│   ├── feature_selection.py                # Script for local feature selection methods
│   └── evaluation.py                       # Script for evaluating federated vs centralized scenarios
├── results/                                # Directory to store logs and evaluation outputs
│   ├── federated_results.txt
│   ├── centralized_results.txt
│   └── plots/
│       ├── performance_comparison.png      # Plots comparing performance metrics
│       └── feature_selection_overlap.png
├── requirements.txt                        # List of dependencies
└── README.md                               # Setup instructions, usage guide, etc.
```

### Requirements
To install the dependencies, run:
```sh
pip install -r requirements.txt
```

### How to Perform the Experiment

#### Step 1: Setting Up the Server
- The server is responsible for aggregating feature rankings from all clients. Start the server by running:
```sh
python src/server.py
```
- This will initialize the server, which listens for incoming feature rankings from clients and aggregates them to determine the final global feature importance.

#### Step 2: Setting Up Clients
- Update the `SERVER_URL` in `client.py` to match the IP address of the server (e.g., `http://<server_ip>:5000`).
- Each client computes feature rankings locally and sends them to the server.
- Run the client script by specifying the client ID:
```sh
python src/client.py
```
- You can run multiple clients on different machines or on the same machine with different `client_id`s to simulate a federated environment.

#### Step 3: Aggregating Feature Rankings
- Once all clients have sent their feature rankings to the server, the server aggregates these rankings and determines the global feature subset.
- Clients can then query the server to get the global feature ranking.

#### Step 4: Centralized Training with Feature Selection
- To compare the federated approach with a centralized approach, perform centralized feature selection using the full dataset.
- Run the following script to perform centralized feature selection and training:
```sh
python src/centralized_feature_selection_training.py
```
- This script uses mutual information to select the most important features and then trains a model using only those features.

#### Step 5: Centralized Training with All Features
- To provide a baseline for comparison, run centralized training using all available features:
```sh
python src/centralized_training_all_features.py
```
- This script trains a model without any feature selection, using all features in the dataset.

#### Step 6: Evaluating the Results
- Use the `evaluation.py` script to generate comparisons between the federated and centralized scenarios.
- This script will output key metrics such as **accuracy**, **F1 score**, **precision**, **recall**, and **AUC** to help understand the performance differences between the approaches.
```sh
python src/evaluation.py
```
- The evaluation results, along with plots, will be saved in the `results/` directory for further analysis.

### Results and Analysis
- The project compares three different training scenarios:
  1. **Federated Feature Selection**: Collaborative feature selection across clients without sharing raw data.
  2. **Centralized Feature Selection**: Feature selection done on the entire dataset in a centralized manner.
  3. **Centralized Training with All Features**: Baseline model trained with all features.
- **Metrics** such as accuracy, precision, recall, F1 score, and AUC are used to evaluate and compare model performance.
- The **results** are stored in the `results/` directory, which includes metrics logs and visual plots for comparison.

### Future Work
- **Improved Aggregation Techniques**: Explore more advanced aggregation techniques (e.g., weighted averaging, consensus methods) for federated feature selection.
- **Non-IID Data**: Test the effectiveness of federated feature selection with non-IID data distributions across clients to evaluate the robustness of the approach.
- **Scalability**: Investigate how the approach scales with an increasing number of clients or larger datasets.

### Contact
If you have any questions or need further assistance, please reach out to the project contributors.