import os
import requests
import pandas as pd
import glob
from typing import List, Dict, Any
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
 
# --- Configuration ---
BASE_URL = os.environ.get("API_BASE_URL", "http://192.168.200.195:9000")
ENDPOINT = "/databricks/table-data-ml"
 
CATALOG_NAME = "lohiya_sales_dev"
SCHEMA_NAME = "gssales"
TABLE_NAME = "fact_secondary_sales"
# Bypass pagination by requesting a massive chunk at once
BATCH_SIZE = 50000
TEST_MODE = False # Will stop after the first batch
 
OUTPUT_DIR = os.path.join("data", "raw")
BATCH_DIR = os.path.join(OUTPUT_DIR, "batches")
FINAL_PARQUET = os.path.join(OUTPUT_DIR, f"{TABLE_NAME}.parquet")
 
def get_session(retries: int = 5, backoff_factor: float = 0.5) -> requests.Session:
    """Creates a requests.Session with robust retry and backoff logic."""
    session = requests.Session()
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session
 
def fetch_and_save_batches():
    """Fetches data with pagination and saves each batch to disk."""
    os.makedirs(BATCH_DIR, exist_ok=True)
    session = get_session()
    headers = {"Accept": "application/json"}
   
    offset = 0
    limit = BATCH_SIZE
   
    print(f"Starting data extraction for {CATALOG_NAME}.{SCHEMA_NAME}.{TABLE_NAME}")
    print(f"Batch size: {limit} | Test mode: {TEST_MODE}")
   
    last_batch_records = None
   
    while True:
        # Saving batches as CSV to avoid pyarrow import issues during extraction
        batch_file = os.path.join(BATCH_DIR, f"batch_{offset}.csv")
       
        # Resume support: skip if the batch already exists
        if os.path.exists(batch_file):
            print(f"Batch at offset {offset} already exists. Skipping download.")
            offset += limit
            if TEST_MODE:
                break
            continue
           
        print(f"Fetching offset {offset}...")
        params = {
            "catalog_name": CATALOG_NAME,
            "schema_name": SCHEMA_NAME,
            "table_name": TABLE_NAME,
            "limit": limit,
            "offset": offset
        }
       
        url = f"{BASE_URL}{ENDPOINT}"
        try:
            # Explicit timeout handling: (30s connect, 600s read) for larger batches
            response = session.get(url, headers=headers, params=params, timeout=(30, 600))
            response.raise_for_status()
           
            data = response.json()
            if isinstance(data, list):
                records = data
            elif isinstance(data, dict):
                records = data.get("data", data.get("items", []))
            else:
                records = []
               
            if not records:
                print("No more records returned. Finished fetching.")
                break
               
            # --- Check if the API is repeating data ---
            if last_batch_records is not None and records == last_batch_records:
                print("All unique data has been extracted. Ending pagination loop.")
                break
           
            last_batch_records = records
           
            # Save batch
            df_batch = pd.DataFrame(records)
            df_batch.to_csv(batch_file, index=False)
            print(f"Saved {len(records)} records to {batch_file}")
           
            # If the backend returned fewer rows than requested, it's the last batch
            if len(records) < limit:
                print("Last partial batch received. Finished fetching.")
                break
               
            offset += limit
           
            if TEST_MODE:
                print("Test mode enabled: stopping after one batch.")
                break
               
        except requests.exceptions.RequestException as e:
            print(f"HTTP Request failed at offset {offset}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response Content: {e.response.text}")
            print("Exiting. You can rerun the script to resume from this point.")
            break
 
def combine_and_save():
    """Combines all downloaded batches, checks for duplicates, and saves to Parquet."""
    print("Combining batches...")
    batch_files = glob.glob(os.path.join(BATCH_DIR, "batch_*.csv"))
    if not batch_files:
        print("No batch files found to combine.")
        return
       
    dfs = []
    for file in batch_files:
        dfs.append(pd.read_csv(file))
       
    final_df = pd.concat(dfs, ignore_index=True)
   
    # Duplicate check
    initial_count = len(final_df)
    final_df = final_df.drop_duplicates()
    dupes_dropped = initial_count - len(final_df)
    if dupes_dropped > 0:
        print(f"Dropped {dupes_dropped} duplicate rows.")
       
    print(f"Saving final dataset to {FINAL_PARQUET}...")
    try:
        # We attempt to save it using fastparquet (or pyarrow)
        final_df.to_parquet(FINAL_PARQUET, index=False)
        final_output_path = FINAL_PARQUET
    except Exception as e:
        print(f"Failed to save as Parquet (due to Application Control policy block?): {e}")
        # Fallback to CSV if Parquet extensions are blocked
        fallback = FINAL_PARQUET.replace(".parquet", ".csv")
        print(f"Saving to CSV fallback: {fallback}")
        final_df.to_csv(fallback, index=False)
        final_output_path = fallback
       
    print("\n--- Process Complete ---")
    print(f"Final row count: {len(final_df)}")
    print(f"Final output path: {os.path.abspath(final_output_path)}")
 
def main():
    # 1. Fetch batches (supports resume and retries)
    fetch_and_save_batches()
    # 2. Combine, deduplicate, and export
    combine_and_save()
 
if __name__ == "__main__":
    main()
 