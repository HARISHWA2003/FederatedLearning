# src/feature_selection.py
from sklearn.feature_selection import mutual_info_classif
import numpy as np

def local_feature_ranking(X, y, num_features=5):
    # Calculate mutual information between each feature and target
    scores = mutual_info_classif(X, y)
    ranking = np.argsort(-scores)[:num_features]
    return ranking, scores

if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X, y = data.data, data.target

    ranking, scores = local_feature_ranking(X, y)
    print(f"Top features (based on mutual information): {ranking}")
