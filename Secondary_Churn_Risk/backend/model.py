import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
import json
import os

# Load config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

def load_data(base_path):
    # Expects fact_secondary_sales.parquet in base_path
    fact_path = os.path.join(base_path, "fact_secondary_sales.parquet")
    
    if not os.path.exists(fact_path):
        # Gracefully handle missing files since we skip fetch data for now
        return pd.DataFrame()
        
    df = pd.read_parquet(fact_path)
    
    # Ensure order_date is datetime and extract month (as period or start of month)
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['month'] = df['order_date'].dt.to_period('M')
    return df

def build_zero_fill_grid(df):
    if df.empty: return df
    
    # Aggregate to monthly grain
    monthly_sales = df.groupby(['outlet_sk', 'sku_sk', 'month']).agg({
        'qty_cases': 'sum',
        'net_amount': 'sum'
    }).reset_index()

    # Create continuous calendar
    min_month = monthly_sales['month'].min()
    max_month = monthly_sales['month'].max()
    all_months = pd.period_range(min_month, max_month, freq='M')

    # Unique outlet-sku combinations
    unique_combos = monthly_sales[['outlet_sk', 'sku_sk']].drop_duplicates()
    
    # Cross join combos with all months
    unique_combos['key'] = 1
    months_df = pd.DataFrame({'month': all_months, 'key': 1})
    grid = pd.merge(unique_combos, months_df, on='key').drop('key', axis=1)

    # Merge actual sales into the grid
    full_grid = pd.merge(grid, monthly_sales, on=['outlet_sk', 'sku_sk', 'month'], how='left')
    full_grid['qty_cases'] = full_grid['qty_cases'].fillna(0)
    full_grid['net_amount'] = full_grid['net_amount'].fillna(0)
    
    return full_grid.sort_values(by=['outlet_sk', 'sku_sk', 'month'])

def engineer_features(df):
    if df.empty: return df
    
    # Group by entity for sequential features
    df = df.copy()
    grouped = df.groupby(['outlet_sk', 'sku_sk'])
    
    # Price = Revenue / Qty (ffill)
    df['price'] = (df['net_amount'] / df['qty_cases']).replace([np.inf, -np.inf], np.nan)
    df['price'] = grouped['price'].ffill().fillna(0) # In case of initial zeros
    
    # Months Inactive counter (Vectorized)
    df['is_zero'] = (df['qty_cases'] == 0).astype(int)
    df['block'] = (df['qty_cases'] > 0).groupby([df['outlet_sk'], df['sku_sk']]).cumsum()
    df['months_inactive'] = df.groupby(['outlet_sk', 'sku_sk', 'block'])['is_zero'].cumsum()
    df.drop(['is_zero', 'block'], axis=1, inplace=True)
    
    # Averages (Vectorized)
    df['hist_avg_qty'] = grouped['qty_cases'].expanding().mean().reset_index(level=[0,1], drop=True).sort_index()
    df['recent_avg_qty'] = grouped['qty_cases'].rolling(window=config['recent_months'], min_periods=1).mean().reset_index(level=[0,1], drop=True).sort_index()
    
    # Trend (Diff from prev month)
    df['trend_qty'] = grouped['qty_cases'].diff().fillna(0)
    
    # Target: Next month qty == 0
    df['next_month_qty'] = grouped['qty_cases'].shift(-1)
    df['churn_target'] = (df['next_month_qty'] == 0).astype(int)
    
    return df

def assign_status(row):
    hist_avg = row['hist_avg_qty']
    recent_avg = row['recent_avg_qty']
    qty = row['qty_cases']
    months_inactive = row.get('months_inactive', 0)
    
    min_active = config['min_active_cases']
    lapse_frac = config['lapse_frac']
    decline_pct = config['decline_pct']
    
    # Customers inactive for 3 months or more are always Lapsed
    if months_inactive >= 3:
        return 'Lapsed'
    # Original Secondary Churn Risk logic based on mathematical drop
    elif (hist_avg >= min_active) and (recent_avg < (lapse_frac * hist_avg)):
        return 'Lapsed'
    elif qty < ((1 - decline_pct) * recent_avg):
        return 'Declining'
    else:
        return 'Healthy'

def train_and_score(df):
    if df.empty: return df
    
    # Features for XGBoost
    features = ['months_inactive', 'price', 'hist_avg_qty', 'recent_avg_qty', 'trend_qty']
    
    # We only train on rows where target is not null (i.e. not the very last month)
    train_data = df[df['next_month_qty'].notnull()].copy()
    predict_data = df[df['next_month_qty'].isnull()].copy() # The very latest month
    
    if len(train_data) > 0:
        X_train = train_data[features]
        y_train = train_data['churn_target']
        
        # XGBoost Classifier with strict fixed random seed
        base_clf = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        
        # Calibrate using Isotonic
        calibrated_clf = CalibratedClassifierCV(estimator=base_clf, method='isotonic', cv=min(3, len(train_data)))
        try:
            calibrated_clf.fit(X_train, y_train)
            
            # Predict on latest month
            if len(predict_data) > 0:
                X_pred = predict_data[features]
                predict_data['churn_probability'] = calibrated_clf.predict_proba(X_pred)[:, 1]
        except Exception:
            # Fallback if too little data for CV
            predict_data['churn_probability'] = 0.0
    else:
        # Fallback if not enough data
        predict_data['churn_probability'] = 0.0
        
    return predict_data

def calculate_value_at_risk(df):
    if df.empty: return df
    
    df['status'] = df.apply(assign_status, axis=1)
    df['base_monthly_value'] = df['hist_avg_qty'] * df['price']
    
    # Financial math for VAR
    df['value_at_risk'] = np.where(
        df['months_inactive'] > 0,
        df['base_monthly_value'] * df['months_inactive'],
        df['base_monthly_value'] * df['churn_probability']
    )
    
    return df

def run_pipeline(base_path):
    df = load_data(base_path)
    if df.empty:
        return [] # Return empty if no data
        
    df_grid = build_zero_fill_grid(df)
    df_features = engineer_features(df_grid)
    df_latest_scored = train_and_score(df_features)
    df_final = calculate_value_at_risk(df_latest_scored)
    
    # Convert to list of dicts for API response
    result = df_final[['outlet_sk', 'sku_sk', 'status', 'churn_probability', 'months_inactive', 'value_at_risk']].copy()
    
    # Sort strictly by churn_probability descending
    result = result.sort_values(by='churn_probability', ascending=False)
    
    return result.to_dict(orient='records')
