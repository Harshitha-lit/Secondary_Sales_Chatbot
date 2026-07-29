import sys
import os
import logging
from typing import Dict, Any, List
import datetime

logger = logging.getLogger(__name__)

# Go up 3 levels to reach Secondary_Agent where churn_tools.py is located
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

try:
    from churn_tools import get_top_churn_risks, get_churn_risk_for_outlet, churn_tools_schema
    from forecast_model_tool import query_forecast_model, forecast_model_tool_schema
except ImportError as e:
    logger.error(f"Failed to import ML tools from parent directory: {e}")
    churn_tools_schema = []
    forecast_model_tool_schema = {}

try:
    from query_data_tool import query_secondary_data, query_data_tool_schema
except ImportError as e:
    logger.error(f"Failed to import query_data_tool: {e}")
    query_data_tool_schema = None

AVAILABLE_TOOLS = churn_tools_schema + ([forecast_model_tool_schema] if forecast_model_tool_schema else []) + ([query_data_tool_schema] if query_data_tool_schema else [])

def make_json_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif hasattr(obj, 'item') and callable(getattr(obj, 'item')):
        return obj.item()
    elif hasattr(obj, 'isoformat') and callable(getattr(obj, 'isoformat')):
        return obj.isoformat()
    elif str(type(obj)) == "<class 'pandas.core.series.Series'>":
        return make_json_serializable(obj.to_dict())
    else:
        return obj

def execute_tool(tool_name: str, arguments: dict) -> Any:
    logger.info(f"Executing ML tool: {tool_name} with args: {arguments}")
    
    try:
        if tool_name == "get_top_churn_risks":
            result = get_top_churn_risks(**arguments)
            return make_json_serializable(result)
        elif tool_name == "get_churn_risk_for_outlet":
            result = get_churn_risk_for_outlet(**arguments)
            return make_json_serializable(result)
        elif tool_name == "query_forecast_model":
            result_str = query_forecast_model(**arguments)
            return {"status": "success", "data": str(result_str)}
        elif tool_name == "query_secondary_data":
            result_str = query_secondary_data(**arguments)
            return {"status": "success", "data": str(result_str)}
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    except Exception as e:
        logger.error(f"Error executing ML tool {tool_name}: {e}")
        return {"error": str(e)}
