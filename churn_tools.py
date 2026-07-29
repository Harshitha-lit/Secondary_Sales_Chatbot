import sys
import os
from typing import List, Dict, Any, Optional

# --- Configuration ---
# Hardcode the absolute path to your Churn_Risk_Prediction project
# so that you can move this tool file anywhere (like your main agent tools folder)
CHURN_PROJECT_PATH = r"C:\Users\LohiyaGroup\Documents\Secondary_Agent\Secondary_Churn_Risk"

# Add the project path and its backend folder to sys.path so we can import the model directly
if CHURN_PROJECT_PATH not in sys.path:
    sys.path.append(CHURN_PROJECT_PATH)
    
BACKEND_PATH = os.path.join(CHURN_PROJECT_PATH, "backend")
if BACKEND_PATH not in sys.path:
    sys.path.append(BACKEND_PATH)

# Now we can import the run_pipeline function directly from your model.py
# without needing the FastAPI backend to be running!
try:
    from model import run_pipeline
except ImportError as e:
    print(f"Error importing model. Make sure CHURN_PROJECT_PATH is correct. Details: {e}")

def get_all_churn_predictions() -> List[Dict[str, Any]]:
    """
    Runs the model pipeline directly (without a separate backend) and returns predictions.
    """
    try:
        # Run the pipeline locally, passing the base directory containing the parquet file
        results = run_pipeline(CHURN_PROJECT_PATH)
        return results
    except Exception as e:
        print(f"Failed to run churn pipeline directly: {e}")
        return []

import pandas as pd

def get_top_churn_risks(limit: int = 10, status_filter: Optional[str] = None, city: Optional[str] = None, zone: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get the top N entities with the highest churn probability.
    """
    data = get_all_churn_predictions()
    
    # Load dimensions to enrich data and allow filtering
    try:
        dim_outlet = pd.read_parquet(os.path.join(CHURN_PROJECT_PATH, 'dim_outlet.parquet'))
        # Create a dictionary mapping outlet_sk to city and zone
        outlet_sk_to_city = dict(zip(dim_outlet['outlet_sk'].astype(str), dim_outlet['city'].fillna("")))
        
        # In dim_outlet, some columns might be missing depending on schema, let's just safely try to get zone if available, else skip
        # Wait, zone is actually in dim_distributor or maybe not in dim_outlet directly, but city is in dim_outlet.
    except Exception as e:
        print(f"Failed to load dimensions for enrichment: {e}")
        outlet_sk_to_city = {}
        
    enriched_data = []
    for row in data:
        osk = str(row.get("outlet_sk"))
        row["city"] = outlet_sk_to_city.get(osk, "")
        enriched_data.append(row)
        
    if status_filter:
        enriched_data = [row for row in enriched_data if row.get("status", "").lower() == status_filter.lower()]
        
    if city:
        enriched_data = [row for row in enriched_data if city.lower() in row.get("city", "").lower()]
        
    # Sort by churn_probability descending, and then by value_at_risk to break ties
    data_sorted = sorted(enriched_data, key=lambda x: (x.get("churn_probability", 0), x.get("value_at_risk", 0)), reverse=True)
    
    return data_sorted[:limit]

def get_churn_risk_for_outlet(outlet_sk: str) -> List[Dict[str, Any]]:
    """
    Get the churn risk and status for a specific outlet.
    """
    data = get_all_churn_predictions()
    return [row for row in data if str(row.get("outlet_sk")) == str(outlet_sk)]

# =====================================================================
# OpenAI / LangChain Tool Definitions
# =====================================================================

churn_tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "get_top_churn_risks",
            "description": "Get the top outlets and SKUs that are at the highest risk of churning based on the churn model.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "The maximum number of top at-risk entities to return. Default is 10."
                    },
                    "status_filter": {
                        "type": "string",
                        "description": "Optional filter for status. Valid options are: 'Lapsed', 'Declining', 'Healthy'."
                    },
                    "city": {
                        "type": "string",
                        "description": "Optional filter to only return churn risks for a specific city."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_churn_risk_for_outlet",
            "description": "Get the churn risk probability, status, and value at risk for a specific outlet by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "outlet_sk": {
                        "type": "string",
                        "description": "The unique identifier (outlet_sk) for the outlet."
                    }
                },
                "required": ["outlet_sk"]
            }
        }
    }
]

if __name__ == "__main__":
    # Test running directly without the backend
    print("Running pipeline directly...")
    top_3 = get_top_churn_risks(limit=3)
    print("Top 3 Risks:", top_3)
