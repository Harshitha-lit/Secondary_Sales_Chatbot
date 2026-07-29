import os
import pandas as pd
import numpy as np

def main():
    input_path = os.path.join("data", "processed", "secondary_monthly_training_panel.parquet")
    output_dir = os.path.join("data", "processed")
    output_path = os.path.join(output_dir, "secondary_feature_dataset.parquet")
    
    print(f"Reading monthly panel from {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"Loaded monthly panel shape: {df.shape}")
    
    # Ensure sorted by group and time
    df = df.sort_values(['distributor_sk', 'sku_sk', 'month']).reset_index(drop=True)
    
    # 1. Reindexing to complete monthly grid per group to avoid shift alignment issues
    print("Building complete monthly grid per group...")
    all_months = pd.date_range(start=df['month'].min(), end=df['month'].max(), freq='MS')
    
    # Get all unique distributor/sku combinations
    keys = df[['distributor_sk', 'sku_sk']].drop_duplicates()
    
    # Multi-index from keys and all_months
    mux = pd.MultiIndex.from_product(
        [keys['distributor_sk'].unique(), keys['sku_sk'].unique(), all_months],
        names=['distributor_sk', 'sku_sk', 'month']
    )
    
    # Reindex df to complete monthly panel
    df_grid = df.set_index(['distributor_sk', 'sku_sk', 'month']).reindex(mux).reset_index()
    
    # Fill missing values for quantity and revenue with 0 (since no sales occurred in those months)
    df_grid['quantity'] = df_grid['quantity'].fillna(0.0)
    df_grid['revenue'] = df_grid['revenue'].fillna(0.0)
    
    # Ensure sorted by group and month before grouping operations
    df_grid = df_grid.sort_values(['distributor_sk', 'sku_sk', 'month']).reset_index(drop=True)
    
    print(f"Grid-completed dataset shape: {df_grid.shape}")
    
    # Group by distributor and SKU
    grouped = df_grid.groupby(['distributor_sk', 'sku_sk'])
    
    print("Creating Lag features (1, 2, 3 months)...")
    df_grid['qty_lag_1'] = grouped['quantity'].shift(1)
    df_grid['qty_lag_2'] = grouped['quantity'].shift(2)
    df_grid['qty_lag_3'] = grouped['quantity'].shift(3)
    
    df_grid['rev_lag_1'] = grouped['revenue'].shift(1)
    df_grid['rev_lag_2'] = grouped['revenue'].shift(2)
    df_grid['rev_lag_3'] = grouped['revenue'].shift(3)
    
    print("Creating Rolling features (2, 3 months)...")
    # Group again on the df_grid using the key columns to do rolling calculations on the lag variables
    # This is fast and vectorized because it avoids python loop overhead of transform(lambda x: ...)
    lag_grouped = df_grid.groupby(['distributor_sk', 'sku_sk'])
    
    df_grid['qty_rolling_mean_2'] = lag_grouped['qty_lag_1'].rolling(window=2, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    df_grid['qty_rolling_mean_3'] = lag_grouped['qty_lag_1'].rolling(window=3, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    df_grid['qty_rolling_std_2'] = lag_grouped['qty_lag_1'].rolling(window=2, min_periods=1).std().reset_index(level=[0, 1], drop=True)
    df_grid['qty_rolling_std_3'] = lag_grouped['qty_lag_1'].rolling(window=3, min_periods=1).std().reset_index(level=[0, 1], drop=True)
    
    df_grid['rev_rolling_mean_2'] = lag_grouped['rev_lag_1'].rolling(window=2, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    df_grid['rev_rolling_mean_3'] = lag_grouped['rev_lag_1'].rolling(window=3, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    df_grid['rev_rolling_std_2'] = lag_grouped['rev_lag_1'].rolling(window=2, min_periods=1).std().reset_index(level=[0, 1], drop=True)
    df_grid['rev_rolling_std_3'] = lag_grouped['rev_lag_1'].rolling(window=3, min_periods=1).std().reset_index(level=[0, 1], drop=True)
    
    print("Creating Growth features...")
    # MoM growth (percentage change between lag 1 and lag 2)
    epsilon = 1e-5
    df_grid['qty_growth_mom'] = (df_grid['qty_lag_1'] - df_grid['qty_lag_2']) / (df_grid['qty_lag_2'] + epsilon)
    df_grid['rev_growth_mom'] = (df_grid['rev_lag_1'] - df_grid['rev_lag_2']) / (df_grid['rev_lag_2'] + epsilon)
    
    print("Creating Calendar and Seasonality features...")
    df_grid['month_val'] = df_grid['month'].dt.month
    df_grid['quarter'] = df_grid['month'].dt.quarter
    
    # Sine/Cosine encoding for seasonality
    df_grid['sin_month'] = np.sin(2 * np.pi * df_grid['month_val'] / 12.0)
    df_grid['cos_month'] = np.cos(2 * np.pi * df_grid['month_val'] / 12.0)
    
    print("Creating Forecast Targets (30, 60, 90-day targets)...")
    # Since granularity is monthly:
    # 30-day target is lead 1 (value of next month)
    # 60-day target is lead 2 (value of 2 months ahead)
    # 90-day target is lead 3 (value of 3 months ahead)
    df_grid['target_qty_30d'] = grouped['quantity'].shift(-1)
    df_grid['target_qty_60d'] = grouped['quantity'].shift(-2)
    df_grid['target_qty_90d'] = grouped['quantity'].shift(-3)
    
    df_grid['target_rev_30d'] = grouped['revenue'].shift(-1)
    df_grid['target_rev_60d'] = grouped['revenue'].shift(-2)
    df_grid['target_rev_90d'] = grouped['revenue'].shift(-3)
    
    # Sort again to ensure clean outputs
    df_grid = df_grid.sort_values(['distributor_sk', 'sku_sk', 'month']).reset_index(drop=True)
    
    # Display columns and head sample
    print(f"\nFinal feature dataset columns: {list(df_grid.columns)}")
    print(f"Shape of feature dataset: {df_grid.shape}")
    print("\nSample rows:")
    print(df_grid[['distributor_sk', 'sku_sk', 'month', 'quantity', 'qty_lag_1', 'qty_rolling_mean_2', 'qty_growth_mom', 'target_qty_30d']].head(10))
    
    # Save the output dataset
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nSaving features dataset to {output_path}...")
    df_grid.to_parquet(output_path, index=False)
    print("Features dataset saved successfully!")

if __name__ == "__main__":
    main()
