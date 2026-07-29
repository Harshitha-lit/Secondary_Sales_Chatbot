import os
import json
import pandas as pd
from typing import Optional

# Path definitions
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Secondary_Forecast")
FORECASTS_PATH = os.path.join(BASE_DIR, "data", "processed", "secondary_forecasts.parquet")
METRICS_PATH = os.path.join(BASE_DIR, "models", "evaluation_metrics.json")

def query_forecast_model(query_type: str, distributor_sk: Optional[int] = None, sku_sk: Optional[int] = None, horizon: Optional[str] = None) -> str:
    """
    Answers questions related to the forecast model, including generated forecasts and model evaluation metrics.
    
    Args:
        query_type (str): Type of question. Options: "get_forecast", "get_metrics", "how_it_works".
        distributor_sk (int, optional): The distributor ID for getting specific forecasts.
        sku_sk (int, optional): The SKU ID for getting specific forecasts.
        horizon (str, optional): The forecast horizon ("30d", "60d", or "90d").
        
    Returns:
        str: The answer or prediction results based on the forecast model.
    """
    if query_type == "how_it_works":
        return (
            "The Forecast Model predicts secondary sales quantities for 30d, 60d, and 90d horizons. "
            "It uses features like historical quantity lags, revenue lags, rolling means, rolling standard deviations, "
            "and seasonal indicators (month_val, quarter, sin_month, cos_month). It is trained using XGBoost / scikit-learn."
        )
        
    elif query_type == "get_metrics":
        try:
            with open(METRICS_PATH, "r") as f:
                metrics = json.load(f)
            # Format metrics nicely
            result = "Forecast Model Evaluation Metrics:\n"
            for target, evals in metrics.items():
                result += f" - {target.replace('target_qty_', '')} Horizon: "
                result += f"MAE = {evals.get('mae', 'N/A'):.2f}, RMSE = {evals.get('rmse', 'N/A'):.2f}, R2 = {evals.get('r2', 'N/A'):.2f}\n"
            return result
        except FileNotFoundError:
            return "Metrics file not found. Please ensure the models have been trained and metrics generated."
            
    elif query_type == "get_forecast":
        try:
            df = pd.read_parquet(FORECASTS_PATH)
            
            # If both are None, provide an overall aggregate forecast
            if distributor_sk is None and sku_sk is None:
                if horizon:
                    if horizon == "30d":
                        df = df[df['horizon'] == "30d"]
                    elif horizon == "60d":
                        df = df[df['horizon'].isin(["30d", "60d"])]
                    elif horizon == "90d":
                        df = df[df['horizon'].isin(["30d", "60d", "90d"])]
                
                overall_sum = df.groupby(['forecast_month'])['predicted_quantity'].sum().reset_index()
                if overall_sum.empty:
                    return "No overall forecast data found."
                    
                total_cumulative = overall_sum['predicted_quantity'].sum()
                result = f"Cumulative Business Forecast for {horizon}:\n"
                result += f" - Total Cumulative Quantity = {total_cumulative:.2f}\n\nMonthly Breakdown:\n"
                
                for _, row in overall_sum.iterrows():
                    result += f" - Month: {row['forecast_month'].strftime('%Y-%m')}: Predicted Quantity = {row['predicted_quantity']:.2f}\n"
                
                # Add Top 10 SKUs breakdown for the specified horizon
                top_skus = df.groupby(['sku_sk'])['predicted_quantity'].sum().reset_index()
                top_skus = top_skus.sort_values(by='predicted_quantity', ascending=False).head(10)
                
                result += "\nTop 10 SKUs Driving This Forecast:\n"
                for _, row in top_skus.iterrows():
                    result += f" - SKU {int(row['sku_sk'])}: {row['predicted_quantity']:.2f}\n"
                    
                return result
                
            # If one is provided but not the other, return an error
            elif distributor_sk is None or sku_sk is None:
                return "Please provide both distributor_sk and sku_sk to retrieve a specific forecast, or leave both empty for the overall forecast."
                
            # Filter the dataframe for the specified distributor and sku
            mask = (df['distributor_sk'] == distributor_sk) & (df['sku_sk'] == sku_sk)
            if horizon:
                mask &= (df['horizon'] == horizon)
                
            filtered = df[mask]
            
            if filtered.empty:
                return f"No forecast data found for distributor_sk: {distributor_sk}, sku_sk: {sku_sk}."
                
            result = f"Forecast for distributor {distributor_sk}, SKU {sku_sk}:\n"
            for _, row in filtered.iterrows():
                result += f" - {row['horizon']} horizon (Month: {row['forecast_month'].strftime('%Y-%m')}): Predicted Quantity = {row['predicted_quantity']:.2f}\n"
            return result
            
        except FileNotFoundError:
            return "Forecast data not found. Please ensure the forecasts have been generated using generate_forecasts.py."
            
    return "Invalid query_type. Please use 'get_forecast', 'get_metrics', or 'how_it_works'."

# -------------------------------------------------------------
# LLM Integration Schemas
# -------------------------------------------------------------

# Example Schema for OpenAI Tool Calling:
forecast_model_tool_schema = {
    "type": "function",
    "function": {
        "name": "query_forecast_model",
        "description": "Get forecasts and model evaluation metrics for the secondary sales forecasting system.",
        "parameters": {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": ["get_forecast", "get_metrics", "how_it_works"],
                    "description": "What to query: 'get_forecast' to retrieve predictions, 'get_metrics' to see model accuracy, 'how_it_works' for details on the model features."
                },
                "distributor_sk": {
                    "type": "integer",
                    "description": "The distributor identifier (required for get_forecast)."
                },
                "sku_sk": {
                    "type": "integer",
                    "description": "The SKU identifier (required for get_forecast)."
                },
                "horizon": {
                    "type": "string",
                    "enum": ["30d", "60d", "90d"],
                    "description": "The forecast horizon. Optional."
                }
            },
            "required": ["query_type"]
        }
    }
}
