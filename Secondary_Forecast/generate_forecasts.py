import os
import pandas as pd
import numpy as np
import joblib

def main():
    input_path = os.path.join("data", "processed", "secondary_feature_dataset.parquet")
    models_dir = "models"
    output_dir = os.path.join("data", "processed")
    output_path = os.path.join(output_dir, "secondary_forecasts.parquet")
    
    print(f"Reading feature dataset from {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"Loaded dataset shape: {df.shape}")
    
    # Identify the latest month to make predictions for future horizons
    latest_month = df['month'].max()
    print(f"Latest available month in historical data: {latest_month}")
    
    # Extract the features for the latest month (inference dataset)
    inference_df = df[df['month'] == latest_month].copy().reset_index(drop=True)
    print(f"Inference dataset size: {len(inference_df)}")
    
    # Features to use for forecasting (must match training features)
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
    
    X_latest = inference_df[features]
    
    horizons = {
        '30d': {'model': 'model_target_qty_30d.joblib', 'offset': 1},
        '60d': {'model': 'model_target_qty_60d.joblib', 'offset': 2},
        '90d': {'model': 'model_target_qty_90d.joblib', 'offset': 3}
    }
    
    all_forecasts = []
    
    for name, config in horizons.items():
        model_name = config['model']
        offset = config['offset']
        model_path = os.path.join(models_dir, model_name)
        
        print(f"Loading model {model_name}...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found. Please train models first.")
            
        model = joblib.load(model_path)
        
        print(f"Generating {name} forecast...")
        # Predict using latest features
        preds = model.predict(X_latest)
        preds = np.maximum(preds, 0.0) # Clip negative predictions to 0
        
        # Calculate future date
        forecast_month = latest_month + pd.DateOffset(months=offset)
        
        # Build forecast dataframe
        forecast_df = inference_df[['distributor_sk', 'sku_sk']].copy()
        forecast_df['forecast_month'] = forecast_month
        forecast_df['predicted_quantity'] = preds
        forecast_df['horizon'] = name
        
        all_forecasts.append(forecast_df)
        
    # Combine predictions
    print("\nCombining forecasts...")
    combined_forecasts = pd.concat(all_forecasts, ignore_index=True)
    combined_forecasts = combined_forecasts.sort_values(by=['distributor_sk', 'sku_sk', 'forecast_month']).reset_index(drop=True)
    
    # Save the output panel
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving combined forecasts to {output_path}...")
    combined_forecasts.to_parquet(output_path, index=False)
    
    print("\n--- Validation & Info ---")
    print(f"Total forecast rows: {len(combined_forecasts)}")
    print(f"Missing values in output: {combined_forecasts.isnull().sum().sum()}")
    print("\nSample predictions:")
    print(combined_forecasts.head(10))

if __name__ == "__main__":
    main()
