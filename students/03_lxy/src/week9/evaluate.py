"""
Module: week9.evaluate
Purpose: Model diagnostics and cross-validation evaluation.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.models import AnalyticalOLS
from utils.diagnostics import calculate_vif


def main():
    # Assume clean data is in data/clean_marketing.csv
    clean_data_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'homework', 'week09', 'data', 'clean_marketing.csv')

    if not os.path.exists(clean_data_path):
        print(f"Error: Clean data file not found at {clean_data_path}")
        print("Please run data_prep.py first to generate clean_marketing.csv")
        sys.exit(1)

    # Load clean data
    df = pd.read_csv(clean_data_path)
    print(f"Loaded clean data with shape: {df.shape}")

    # Assume Sales is target, others are features
    target_col = 'Sales'
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found in data.")
        sys.exit(1)

    y = df[target_col].values
    X = df.drop(columns=[target_col]).values
    feature_names = df.drop(columns=[target_col]).columns.tolist()

    print(f"Features: {feature_names}")

    # Multicollinearity check
    vif_values = calculate_vif(X)
    high_vif_features = []
    for i, vif in enumerate(vif_values):
        if vif > 10:
            high_vif_features.append((feature_names[i], vif))

    if high_vif_features:
        print("\033[91mWARNING: High multicollinearity detected!\033[0m")  # Red text
        for feat, vif_val in high_vif_features:
            print(f"Feature '{feat}' has VIF = {vif_val:.2f} (> 10)")
        print("This may indicate multicollinearity between features.")
        # To identify pairs, we can compute correlation matrix
        corr_matrix = np.corrcoef(X.T)
        for i in range(len(feature_names)):
            for j in range(i+1, len(feature_names)):
                if abs(corr_matrix[i, j]) > 0.8:  # High correlation threshold
                    print(f"High correlation between '{feature_names[i]}' and '{feature_names[j]}': {corr_matrix[i,j]:.2f}")
    else:
        print("No high multicollinearity detected (all VIF <= 10).")

    # 5-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = AnalyticalOLS()
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            r2 = r2_score(y_val, y_pred)
            r2_scores.append(r2)
            print(f"Fold {fold+1}: R² = {r2:.4f}")
        except np.linalg.LinAlgError as e:
            print(f"Fold {fold+1}: Singular matrix error - {e}")
            r2_scores.append(np.nan)

    valid_scores = [s for s in r2_scores if not np.isnan(s)]
    if valid_scores:
        mean_r2 = np.mean(valid_scores)
        print(f"\nAverage R² across {len(valid_scores)} folds: {mean_r2:.4f}")
    else:
        print("All folds failed due to singular matrix.")


if __name__ == "__main__":
    main()