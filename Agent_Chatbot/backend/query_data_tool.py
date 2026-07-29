import os
import sqlite3
import pandas as pd
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Global connection to keep data cached in memory
_con = None

def _initialize_db():
    global _con
    if _con is not None:
        return _con
        
    logger.info("Initializing in-memory SQLite database for historical data...")
    _con = sqlite3.connect(':memory:', check_same_thread=False)
    
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'Secondary_Churn_Risk')
    
    try:
        for f in os.listdir(base_path):
            if f.endswith('.parquet'):
                table_name = f.replace('.parquet', '')
                df = pd.read_parquet(os.path.join(base_path, f))
                df.to_sql(table_name, _con, index=False, if_exists='replace')
        logger.info("Successfully loaded parquet files into in-memory SQLite.")
    except Exception as e:
        logger.error(f"Failed to load parquet data into SQLite: {e}")
        
    return _con

def query_secondary_data(sql_query: str) -> str:
    """
    Executes a SQL query against the historical secondary sales database.
    """
    logger.info(f"Executing SQL query: {sql_query}")
    try:
        con = _initialize_db()
        df_res = pd.read_sql_query(sql_query, con)
        
        # Limit rows to avoid huge LLM context overflow
        if len(df_res) > 50:
            df_res = df_res.head(50)
            warning = "\nNote: Results were truncated to the first 50 rows."
        else:
            warning = ""
            
        return df_res.to_string(index=False) + warning
    except Exception as e:
        return f"Error executing SQL: {e}"

query_data_tool_schema = {
    "type": "function",
    "function": {
        "name": "query_secondary_data",
        "description": "Query historical secondary sales data using SQLite. \nTABLE SCHEMAS:\n- fact_secondary_sales: outlet_sk, distributor_sk, sku_sk, source_year, source_month, zone_name, state_name, order_date, total_quantity, qty_cases, net_amount, ex_fact_amount\n- dim_distributor: distributor_sk, warehouse_name (distributor name), wd_name, warehouse_city, zone_name, state_name\n- dim_outlet: outlet_sk, outlet_name, city, state_name, outlet_type\n- dim_sku: sku_sk, sku_name (or sku_description), brand_name, category_name\nJOIN these tables using distributor_sk, outlet_sk, or sku_sk.",
        "parameters": {
            "type": "object",
            "properties": {
                "sql_query": {
                    "type": "string",
                    "description": "The exact SQLite query to run against the database. Remember that dates might need string manipulation or extraction. Be mindful of joining dimension tables using outlet_sk, sku_sk, or distributor_sk."
                }
            },
            "required": ["sql_query"]
        }
    }
}
