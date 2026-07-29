import os
import json
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import joblib

def wape(y_true, y_pred):
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))) if np.sum(np.abs(y_true)) > 0 else 0.0

def main():
    input_path = os.path.join("data", "processed", "secondary_feature_dataset.parquet")
    models_dir = "models"
    metrics_path = os.path.join(models_dir, "evaluation_metrics.json")
    
    # Ensure models directory exists
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"Reading feature dataset from {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"Loaded feature dataset shape: {df.shape}")
    
    # Features to use for training
    features = [
        'qty_lag_1', 'qty_lag_2', 'qty_lag_3',
        'rev_lag_1', 'rev_lag_2', 'rev_lag_3',
        'qty_rolling_mean_2', 'qty_rolling_mean_3',
        'qty_rolling_std_2', 'qty_rolling_std_3',
        'rev_rolling_mean_2', 'rev_rolling_mean_3',
        'rev_rolling_std_2', 'rev_rolling_std_3',
        'qty_growth_mom', 'rev_growth_mom',
        'month_val', 'quarter', 'sin_month', 'cos_month'
    ]
    
    targets = ['target_qty_30d', 'target_qty_60d', 'target_qty_90d']
    metrics_summary = {}
    
    for target in targets:
        print(f"\n--- Training LightGBM Tweedie Model for Target: {target} ---")
        
        # Sort data chronologically to ensure TimeSeriesSplit is valid (as in Primary Sales)
        df = df.sort_values(by=['month', 'distributor_sk', 'sku_sk']).reset_index(drop=True)
        
        # Filter for rows where target is not null
        df_target = df.dropna(subset=[target]).copy()
        
        if len(df_target) == 0:
            print(f"Skipping {target} as there are no rows with non-null targets.")
            continue
            
        X = df_target[features]
        y = df_target[target]
        
        print(f"Total available rows: {len(X)}")
        
        # Cross validation using TimeSeriesSplit (3 splits)
        tscv = TimeSeriesSplit(n_splits=3)
        
        cv_wape = []
        cv_rmse = []
        cv_mae = []
        cv_mape = []
        cv_r2 = []
        
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # Train LightGBM Tweedie Regressor
            model = LGBMRegressor(objective='tweedie', random_state=42, n_jobs=-1, verbose=-1)
            model.fit(X_train, y_train)
            
            # Predict and evaluate
            preds = model.predict(X_test)
            preds = np.maximum(preds, 0.0) # Ensure no negative predictions (Tweedie is non-negative)
            
            y_test_mape = np.where(y_test == 0, 1e-5, y_test)
            
            cv_wape.append(wape(y_test, preds))
            cv_rmse.append(float(np.sqrt(mean_squared_error(y_test, preds))))
            cv_mae.append(float(mean_absolute_error(y_test, preds)))
            cv_mape.append(float(mean_absolute_percentage_error(y_test_mape, preds)))
            cv_r2.append(float(r2_score(y_test, preds)))
            
        mean_wape = float(np.mean(cv_wape))
        mean_rmse = float(np.mean(cv_rmse))
        mean_mae = float(np.mean(cv_mae))
        mean_mape = float(np.mean(cv_mape))
        mean_r2 = float(np.mean(cv_r2))
        
        print(f"CV Metrics: WAPE={mean_wape:.4f}, RMSE={mean_rmse:.2f}, MAE={mean_mae:.2f}, MAPE={mean_mape:.2%}, R2={mean_r2:.4f}")
        
        # Train final model on all valid data
        final_model = LGBMRegressor(objective='tweedie', random_state=42, n_jobs=-1, verbose=-1)
        final_model.fit(X, y)
        
        # Save model
        model_filename = f"model_{target}.joblib"
        model_path = os.path.join(models_dir, model_filename)
        joblib.dump(final_model, model_path)
        print(f"Saved final model to {model_path}")
        
        # Store metrics
        metrics_summary[target] = {
            "wape": mean_wape,
            "rmse": mean_rmse,
            "mae": mean_mae,
            "mape": mean_mape,
            "r2": mean_r2,
            "dataset_size": len(X)
        }
        
    # Save overall metrics summary to JSON
    print(f"\nSaving all evaluation metrics to {metrics_path}...")
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
    print("Metrics saved successfully!")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()
