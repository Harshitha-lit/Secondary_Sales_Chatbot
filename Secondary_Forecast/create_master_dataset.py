import os
import pandas as pd
import numpy as np

def detect_join_key(df_fact, df_dim, dim_name):
    """
    Dynamically detects the correct join key between the fact table and a dimension table.
    Looks for common columns, prioritizing those ending with '_sk' and unique in the dimension.
    """
    common_cols = set(df_fact.columns).intersection(set(df_dim.columns))
    
    # 1. Look for common columns ending with '_sk'
    sk_cols = [c for c in common_cols if c.endswith('_sk')]
    if len(sk_cols) == 1:
        print(f"Detected join key '{sk_cols[0]}' for {dim_name} (common column ending in '_sk')")
        return sk_cols[0]
    elif len(sk_cols) > 1:
        # Prioritize the one unique in dimension table
        unique_sk_cols = [c for c in sk_cols if df_dim[c].is_unique]
        if len(unique_sk_cols) == 1:
            print(f"Detected join key '{unique_sk_cols[0]}' for {dim_name} (unique common column ending in '_sk')")
            return unique_sk_cols[0]
        print(f"Detected join key '{sk_cols[0]}' for {dim_name} (first common column ending in '_sk')")
        return sk_cols[0]
        
    # 2. Look for common columns ending in '_id', 'key', or '_code'
    id_cols = [c for c in common_cols if c.endswith('_id') or c.endswith('_code') or c.endswith('key')]
    unique_id_cols = [c for c in id_cols if df_dim[c].is_unique]
    if len(unique_id_cols) == 1:
        print(f"Detected join key '{unique_id_cols[0]}' for {dim_name} (unique common column ending in id/code/key)")
        return unique_id_cols[0]
    elif len(id_cols) >= 1:
        print(f"Detected join key '{id_cols[0]}' for {dim_name} (first common column ending in id/code/key)")
        return id_cols[0]
        
    # 3. Look for any common column unique in the dimension table
    unique_common_cols = [c for c in common_cols if df_dim[c].is_unique]
    if len(unique_common_cols) == 1:
        print(f"Detected join key '{unique_common_cols[0]}' for {dim_name} (only unique common column)")
        return unique_common_cols[0]
        
    if not common_cols:
        raise ValueError(f"No common columns found between fact table and {dim_name} to join on.")
        
    # Fallback to the first common column
    fallback_col = list(common_cols)[0]
    print(f"WARNING: Fallback detected join key '{fallback_col}' for {dim_name}")
    return fallback_col

def main():
    # Define file paths
    fact_path = "fact_secondary_sales.parquet"
    dist_path = "dim_distributor.parquet"
    outlet_path = "dim_outlet.parquet"
    sku_path = "dim_sku.parquet"
    
    output_dir = os.path.join("data", "processed")
    output_path = os.path.join(output_dir, "master_secondary_sales.parquet")
    
    print("Loading datasets...")
    df_fact = pd.read_parquet(fact_path)
    df_dist = pd.read_parquet(dist_path)
    df_outlet = pd.read_parquet(outlet_path)
    df_sku = pd.read_parquet(sku_path)
    
    print(f"Fact table shape: {df_fact.shape}")
    print(f"Distributor dimension shape: {df_dist.shape}")
    print(f"Outlet dimension shape: {df_outlet.shape}")
    print(f"SKU dimension shape: {df_sku.shape}")
    
    # Dynamically detect keys
    print("\n--- Detecting Join Keys ---")
    dist_key = detect_join_key(df_fact, df_dist, "dim_distributor")
    outlet_key = detect_join_key(df_fact, df_outlet, "dim_outlet")
    sku_key = detect_join_key(df_fact, df_sku, "dim_sku")
    
    # Check uniqueness of keys in dimensions
    for name, df, key in [("Distributor", df_dist, dist_key), 
                          ("Outlet", df_outlet, outlet_key), 
                          ("SKU", df_sku, sku_key)]:
        is_unique = df[key].is_unique
        print(f"{name} key '{key}' is unique in dimension table: {is_unique}")
        if not is_unique:
            duplicate_count = df[key].duplicated().sum()
            print(f"  WARNING: {name} table has {duplicate_count} duplicate keys! This may cause row explosion.")

    # List of joins to perform
    joins = [
        {"name": "distributor", "df_dim": df_dist, "key": dist_key, "suffix": "_dist"},
        {"name": "outlet", "df_dim": df_outlet, "key": outlet_key, "suffix": "_outlet"},
        {"name": "sku", "df_dim": df_sku, "key": sku_key, "suffix": "_sku"}
    ]
    
    current_df = df_fact.copy()
    
    for join_info in joins:
        name = join_info["name"]
        df_dim = join_info["df_dim"]
        key = join_info["key"]
        suffix = join_info["suffix"]
        
        row_count_before = len(current_df)
        print(f"\n--- Joining {name} on '{key}' ---")
        print(f"Row count before join: {row_count_before}")
        
        # Identify unmatched keys before join
        fact_keys = set(current_df[key].dropna().unique())
        dim_keys = set(df_dim[key].dropna().unique())
        unmatched_keys = fact_keys - dim_keys
        
        # Calculate stats for unmatched keys in the current_df
        unmatched_rows = current_df[current_df[key].isin(unmatched_keys)]
        unmatched_rows_count = len(unmatched_rows)
        missing_key_rows = current_df[key].isna().sum()
        
        print(f"Unique keys in Fact: {len(fact_keys)}")
        print(f"Unique keys in Dimension: {len(dim_keys)}")
        print(f"Unmatched unique keys: {len(unmatched_keys)}")
        print(f"Unmatched rows in Fact: {unmatched_rows_count} ({unmatched_rows_count/row_count_before:.2%})")
        print(f"Rows with null/missing key in Fact: {missing_key_rows} ({missing_key_rows/row_count_before:.2%})")
        if len(unmatched_keys) > 0:
            print(f"Sample unmatched keys: {list(unmatched_keys)[:10]}")
            
        # Rename overlapping columns in dimension to avoid name collisions (except the join key)
        overlapping_cols = [col for col in df_dim.columns if col in current_df.columns and col != key]
        rename_dict = {col: f"{col}{suffix}" for col in overlapping_cols}
        if rename_dict:
            print(f"Renaming overlapping columns in dim: {rename_dict}")
            df_dim_renamed = df_dim.rename(columns=rename_dict)
        else:
            df_dim_renamed = df_dim
            
        # Perform left join
        current_df = pd.merge(current_df, df_dim_renamed, on=key, how="left")
        
        row_count_after = len(current_df)
        print(f"Row count after join: {row_count_after}")
        if row_count_after != row_count_before:
            print(f"WARNING: Row count changed from {row_count_before} to {row_count_after} (Diff: {row_count_after - row_count_before})")
        else:
            print("Row count validation passed (no change in row count).")
            
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving final master dataset to {output_path}...")
    current_df.to_parquet(output_path, index=False)
    print(f"Master dataset saved successfully with shape {current_df.shape}!")

if __name__ == "__main__":
    main()
