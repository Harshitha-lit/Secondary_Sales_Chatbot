import os
import pandas as pd
import numpy as np

def detect_panel_columns(df):
    """
    Dynamically identifies the correct columns for building the monthly panel.
    """
    cols = list(df.columns)
    
    # 1. Detect Date Column
    date_col = None
    # Prioritize 'order_date' or other columns containing 'date' but not 'loaded_at'
    date_candidates = [c for c in cols if 'date' in c.lower() and 'loaded' not in c.lower()]
    if 'order_date' in date_candidates:
        date_col = 'order_date'
    elif date_candidates:
        date_col = date_candidates[0]
    else:
        raise ValueError("Could not detect date column. Candidates: " + str(date_candidates))
        
    # 2. Detect Distributor Column
    dist_col = None
    if 'distributor_sk' in cols:
        dist_col = 'distributor_sk'
    else:
        dist_candidates = [c for c in cols if 'distributor' in c.lower()]
        if dist_candidates:
            dist_col = dist_candidates[0]
        else:
            raise ValueError("Could not detect distributor column.")
            
    # 3. Detect Product / SKU Column
    prod_col = None
    if 'sku_sk' in cols:
        prod_col = 'sku_sk'
    else:
        prod_candidates = [c for c in cols if 'sku' in c.lower()]
        if prod_candidates:
            prod_col = prod_candidates[0]
        else:
            raise ValueError("Could not detect product/SKU column.")
            
    # 4. Detect Quantity Column
    qty_col = None
    if 'total_quantity' in cols:
        qty_col = 'total_quantity'
    elif 'qty_cases' in cols:
        qty_col = 'qty_cases'
    else:
        qty_candidates = [c for c in cols if 'qty' in c.lower() or 'quantity' in c.lower()]
        if qty_candidates:
            qty_col = qty_candidates[0]
        else:
            raise ValueError("Could not detect quantity column.")
            
    # 5. Detect Revenue Column
    rev_col = None
    # Prioritize net_amount or net_rev_amt
    for candidate in ['net_amount', 'net_rev_amt', 'fulfilled_amount']:
        if candidate in cols:
            rev_col = candidate
            break
    if not rev_col:
        rev_candidates = [c for c in cols if 'amount' in c.lower() or 'rev' in c.lower() or 'sales' in c.lower()]
        if rev_candidates:
            rev_col = rev_candidates[0]
        else:
            raise ValueError("Could not detect revenue column.")
            
    print(f"Dynamic Column Selection Results:")
    print(f"  - Date Column:        '{date_col}'")
    print(f"  - Distributor Column: '{dist_col}'")
    print(f"  - Product (SKU) Column: '{prod_col}'")
    print(f"  - Quantity Column:    '{qty_col}'")
    print(f"  - Revenue Column:     '{rev_col}'")
    
    return date_col, dist_col, prod_col, qty_col, rev_col

def main():
    input_path = os.path.join("data", "processed", "master_secondary_sales.parquet")
    output_dir = os.path.join("data", "processed")
    output_path = os.path.join(output_dir, "secondary_monthly_training_panel.parquet")
    
    print(f"Reading master secondary sales dataset from {input_path}...")
    df = pd.read_parquet(input_path)
    print(f"Loaded master dataset with shape {df.shape}")
    
    # Detect the correct columns
    date_col, dist_col, prod_col, qty_col, rev_col = detect_panel_columns(df)
    
    # Convert date column to datetime
    print("Parsing date column and converting to month-start...")
    df['parsed_date'] = pd.to_datetime(df[date_col])
    # Extract month start
    df['month'] = df['parsed_date'].dt.to_period('M').dt.to_timestamp()
    
    # Aggregate monthly
    print(f"Aggregating data monthly by '{dist_col}', '{prod_col}', and 'month'...")
    monthly_df = df.groupby([dist_col, prod_col, 'month']).agg(
        quantity=(qty_col, 'sum'),
        revenue=(rev_col, 'sum')
    ).reset_index()
    
    print(f"Aggregated dataset shape: {monthly_df.shape}")
    print(monthly_df.head())
    
    # Save the output panel
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving monthly training panel to {output_path}...")
    monthly_df.to_parquet(output_path, index=False)
    print("Monthly training panel saved successfully!")

if __name__ == "__main__":
    main()
