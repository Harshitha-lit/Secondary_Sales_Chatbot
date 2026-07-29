from contextlib import asynccontextmanager
import json
import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Paths and in-memory cache
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
forecast_cache: dict = {}


def resolve_path(env_name: str, default_relative_path: str) -> Path:
    """
    Resolve a path from an environment variable or from the project directory.
    """
    configured_path = os.getenv(env_name, default_relative_path)
    path = Path(configured_path)

    if not path.is_absolute():
        path = BASE_DIR / path

    return path.resolve()


def find_existing_sales_file() -> Optional[Path]:
    """
    Try the configured SALES_PATH first, then common project locations.
    """
    candidates = [
        resolve_path("SALES_PATH", "data/processed/secondary_monthly_training_panel.parquet"),
        BASE_DIR / "data" / "processed" / "secondary_monthly_training_panel.parquet",
        BASE_DIR / "secondary_monthly_training_panel.parquet",
    ]

    for path in candidates:
        if path.exists():
            return path.resolve()

    return None


def clean_numeric(value) -> float:
    """
    Convert a numeric value into a JSON-safe float.
    """
    if value is None or pd.isna(value):
        return 0.0

    value = float(value)

    if not np.isfinite(value):
        return 0.0

    return value


# -----------------------------------------------------------------------------
# Application startup and shutdown
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    forecast_cache.clear()

    # Load raw sales data for the overall revenue history endpoint.
    sales_path = find_existing_sales_file()

    if sales_path is not None:
        try:
            sales_df = pd.read_parquet(sales_path)
            forecast_cache["sales"] = sales_df
            forecast_cache["sales_path"] = str(sales_path)

            logger.info(
                "Loaded sales data from %s with %s rows.",
                sales_path,
                len(sales_df),
            )
        except Exception as exc:
            logger.exception("Unable to load sales data: %s", exc)
    else:
        logger.warning(
            "secondary_monthly_training_panel.parquet was not found. Set SALES_PATH or place the file "
            "in the project root, data/raw, data/processed, or data."
        )

    # Load generated forecast output.
    forecast_path = resolve_path(
        "FORECAST_PATH",
        "data/processed/secondary_forecasts.parquet",
    )

    if forecast_path.exists():
        try:
            forecast_df = pd.read_parquet(forecast_path)

            required_columns = {
                "distributor_sk",
                "sku_sk",
                "forecast_month",
                "horizon",
                "predicted_quantity",
            }
            missing_columns = required_columns.difference(forecast_df.columns)

            if missing_columns:
                raise ValueError(
                    "Forecast file is missing required columns: "
                    + ", ".join(sorted(missing_columns))
                )

            # Map the Secondary Sales columns to API columns
            forecast_df["payer_code"] = forecast_df["distributor_sk"].astype(str)
            forecast_df["pack_type"] = forecast_df["sku_sk"].astype(str)
            forecast_df["predicted_revenue"] = forecast_df["predicted_quantity"]
            forecast_df["P10"] = forecast_df["predicted_quantity"]
            forecast_df["P50"] = forecast_df["predicted_quantity"]
            forecast_df["P90"] = forecast_df["predicted_quantity"]

            forecast_df["forecast_month"] = pd.to_datetime(
                forecast_df["forecast_month"],
                errors="coerce",
            ).dt.strftime("%Y-%m-%d")

            numeric_columns = ["predicted_revenue", "P10", "P50", "P90"]
            for column in numeric_columns:
                forecast_df[column] = pd.to_numeric(
                    forecast_df[column],
                    errors="coerce",
                )

            forecast_cache["data"] = forecast_df
            forecast_cache["forecast_path"] = str(forecast_path)
            forecast_cache["total_series"] = len(
                forecast_df[["payer_code", "pack_type"]].drop_duplicates()
            )

            months = sorted(
                month
                for month in forecast_df["forecast_month"].dropna().unique()
                if month
            )
            forecast_cache["date_range"] = (
                (months[0], months[-1]) if months else (None, None)
            )

            logger.info(
                "Loaded forecast data from %s with %s rows.",
                forecast_path,
                len(forecast_df),
            )
        except Exception as exc:
            logger.exception("Unable to load forecast data: %s", exc)
    else:
        logger.warning("Forecast file not found at %s", forecast_path)

    yield

    forecast_cache.clear()


# -----------------------------------------------------------------------------
# FastAPI application
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Forecast API",
    description="API for customer-level and overall business revenue forecasts",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Response models
# -----------------------------------------------------------------------------
class ForecastRecord(BaseModel):
    payer_code: str
    pack_type: str
    forecast_month: str
    horizon_days: int
    predicted_revenue: float
    P10: float
    P50: float
    P90: float


class StatusResponse(BaseModel):
    status: str
    model_version: str
    latest_data_month: Optional[str]
    forecast_date_range: tuple
    total_series: int


class HealthResponse(BaseModel):
    status: str
    forecast_file_exists: bool
    sales_file_exists: bool
    model_files_exist: bool
    data_valid: bool
    errors: List[str]


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def horizon_to_days(value) -> int:
    """
    Convert values such as '30d' into 30.
    """
    if isinstance(value, str):
        try:
            return int(value.replace("d", "").split("_")[0])
        except (ValueError, IndexError):
            return 0

    if isinstance(value, (int, np.integer)):
        return int(value)

    return 0


def get_revenue_column(df: pd.DataFrame) -> str:
    """
    Return the first valid revenue column found in the sales dataset.
    """
    candidates = [
        "revenue",
        "net_rev_amt",
        "net_revenue",
        "net_value",
        "gross_inv_amt",
        "gross_inv_value",
        "gst_assessable_value",
    ]

    revenue_column = next(
        (column for column in candidates if column in df.columns),
        None,
    )

    if revenue_column is None:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "No supported revenue column was found.",
                "expected_any_of": candidates,
                "available_columns": list(df.columns),
            },
        )

    return revenue_column


def get_date_column(df: pd.DataFrame) -> str:
    """
    Return a usable date column from the sales dataset.
    """
    candidates = ["month", "inv_date", "sales_month", "snapshot_date", "date"]

    date_column = next(
        (column for column in candidates if column in df.columns),
        None,
    )

    if date_column is None:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "No supported date column was found.",
                "expected_any_of": candidates,
                "available_columns": list(df.columns),
            },
        )

    return date_column


def forecast_dataframe_to_records(df: pd.DataFrame) -> List[dict]:
    """
    Convert forecast dataframe rows into the API response format.
    """
    records: List[dict] = []

    for _, row in df.iterrows():
        records.append(
            {
                "payer_code": str(row["payer_code"]),
                "pack_type": str(row["pack_type"]),
                "forecast_month": str(row.get("forecast_month") or ""),
                "horizon_days": horizon_to_days(row.get("horizon")),
                "predicted_revenue": clean_numeric(row.get("predicted_revenue")),
                "P10": clean_numeric(row.get("P10")),
                "P50": clean_numeric(row.get("P50")),
                "P90": clean_numeric(row.get("P90")),
            }
        )

    return records


# -----------------------------------------------------------------------------
# Basic routes
# -----------------------------------------------------------------------------
@app.get("/")
def root():
    return {"message": "Forecast API is running"}


@app.get("/api")
def api_info():
    return {
        "message": "Forecast API is running",
        "docs": "/docs",
        "customer_forecast": "/api/forecast",
        "overall_forecast": "/api/forecast/overall",
        "revenue_history": "/api/revenue/history",
    }


# -----------------------------------------------------------------------------
# Customer-level forecast
# -----------------------------------------------------------------------------
@app.get("/api/forecast", response_model=List[ForecastRecord])
def get_forecasts(
    horizon: Optional[int] = Query(
        None,
        description="Forecast horizon in days: 30, 60, or 90",
    ),
    payer_code: Optional[str] = Query(
        None,
        description="Filter by payer code (mapped to distributor_sk)",
    ),
    pack_type: Optional[str] = Query(
        None,
        description="Filter by pack type (mapped to sku_sk)",
    ),
):
    if "data" not in forecast_cache:
        raise HTTPException(
            status_code=503,
            detail=(
                "Forecast data is not loaded. Verify FORECAST_PATH and restart "
                "the backend."
            ),
        )

    df = forecast_cache["data"].copy()

    if horizon is not None:
        if horizon not in {30, 60, 90}:
            raise HTTPException(
                status_code=400,
                detail="Horizon must be 30, 60, or 90.",
            )

        df = df[df["horizon"].apply(horizon_to_days) == horizon]

    if payer_code is not None:
        df = df[df["payer_code"].astype(str) == str(payer_code)]

    if pack_type is not None:
        requested_pack_type = str(pack_type).strip().casefold()
        df = df[
            df["pack_type"]
            .astype(str)
            .str.strip()
            .str.casefold()
            .eq(requested_pack_type)
        ]

    return forecast_dataframe_to_records(df)


# -----------------------------------------------------------------------------
# Model status, metrics, and health
# -----------------------------------------------------------------------------
@app.get("/api/models/forecast/status", response_model=StatusResponse)
def get_status():
    if "data" not in forecast_cache:
        raise HTTPException(
            status_code=503,
            detail="Forecast data is not loaded.",
        )

    date_range = forecast_cache.get("date_range", (None, None))

    return {
        "status": "trained",
        "model_version": "v1.1-tweedie",
        "latest_data_month": date_range[0],
        "forecast_date_range": date_range,
        "total_series": forecast_cache.get("total_series", 0),
    }


@app.get("/api/models/forecast/metrics")
def get_metrics():
    metrics_path = resolve_path(
        "METRICS_PATH",
        "models/evaluation_metrics.json",
    )

    if not metrics_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Metrics file not found at {metrics_path}",
        )

    try:
        with metrics_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Metrics file contains invalid JSON: {exc}",
        ) from exc


@app.get("/api/models/forecast/health", response_model=HealthResponse)
def get_health():
    errors: List[str] = []

    forecast_path = resolve_path(
        "FORECAST_PATH",
        "data/processed/secondary_forecasts.parquet",
    )
    forecast_exists = forecast_path.exists()

    sales_path = find_existing_sales_file()
    sales_exists = sales_path is not None

    models_dir = resolve_path("MODELS_DIR", "models")
    missing_model_files = [
        str(models_dir / f"model_target_qty_{horizon}d.joblib")
        for horizon in [30, 60, 90]
        if not (models_dir / f"model_target_qty_{horizon}d.joblib").exists()
    ]
    models_exist = len(missing_model_files) == 0

    if not forecast_exists:
        errors.append(f"Forecast file is missing: {forecast_path}")

    if not sales_exists:
        errors.append(
            "Sales file is missing. Set SALES_PATH or place secondary_monthly_training_panel.parquet "
            "in a supported project folder."
        )

    if missing_model_files:
        errors.extend(
            f"Model file is missing: {path}" for path in missing_model_files
        )

    data_valid = "data" in forecast_cache and not forecast_cache["data"].empty

    if not data_valid:
        errors.append("Forecast data is not loaded or is empty.")

    status = (
        "healthy"
        if forecast_exists and sales_exists and models_exist and data_valid
        else "unhealthy"
    )

    return {
        "status": status,
        "forecast_file_exists": forecast_exists,
        "sales_file_exists": sales_exists,
        "model_files_exist": models_exist,
        "data_valid": data_valid,
        "errors": errors,
    }


# -----------------------------------------------------------------------------
# Historical overall business revenue
# -----------------------------------------------------------------------------
@app.get("/api/revenue/history")
def get_revenue_history():
    if "sales" not in forecast_cache:
        raise HTTPException(
            status_code=503,
            detail=(
                "Sales data is not loaded. Verify SALES_PATH and restart "
                "the backend."
            ),
        )

    sales_df = forecast_cache["sales"].copy()

    date_column = get_date_column(sales_df)
    revenue_column = get_revenue_column(sales_df)

    sales_df["date"] = pd.to_datetime(
        sales_df[date_column],
        errors="coerce",
    )
    sales_df[revenue_column] = pd.to_numeric(
        sales_df[revenue_column],
        errors="coerce",
    )

    sales_df = sales_df.dropna(subset=["date", revenue_column])

    if sales_df.empty:
        raise HTTPException(
            status_code=500,
            detail="No valid dated revenue rows were found in the sales file.",
        )

    sales_df["month_str"] = sales_df["date"].dt.to_period("M").astype(str)

    monthly = (
        sales_df.groupby("month_str", as_index=False)[revenue_column]
        .sum()
        .sort_values("month_str")
        .rename(columns={revenue_column: "revenue", "month_str": "month"})
    )

    monthly_history = [
        {
            "month": str(row["month"]),
            "revenue": clean_numeric(row["revenue"]),
        }
        for _, row in monthly.iterrows()
    ]

    total_historical_revenue = clean_numeric(sales_df[revenue_column].sum())

    latest_month_revenue = None
    previous_month_revenue = None
    mom_growth_percent = None

    if len(monthly) >= 1:
        latest_month_revenue = clean_numeric(monthly.iloc[-1]["revenue"])

    if len(monthly) >= 2:
        previous_month_revenue = clean_numeric(monthly.iloc[-2]["revenue"])

        if previous_month_revenue != 0:
            mom_growth_percent = (
                (latest_month_revenue - previous_month_revenue)
                / previous_month_revenue
                * 100
            )

    return {
        "monthly_history": monthly_history,
        "total_historical_revenue": total_historical_revenue,
        "latest_month_revenue": latest_month_revenue,
        "previous_month_revenue": previous_month_revenue,
        "mom_growth_percent": mom_growth_percent,
        "revenue_column_used": revenue_column,
        "sales_file_used": forecast_cache.get("sales_path"),
    }


# -----------------------------------------------------------------------------
# Overall business forecast
# -----------------------------------------------------------------------------
@app.get("/api/forecast/overall", response_model=List[ForecastRecord])
def get_overall_forecast():
    if "data" not in forecast_cache:
        raise HTTPException(
            status_code=503,
            detail=(
                "Forecast data is not loaded. Verify FORECAST_PATH and restart "
                "the backend."
            ),
        )

    df = forecast_cache["data"].copy()
    df["horizon_days"] = df["horizon"].apply(horizon_to_days)

    for column in ["P10", "P50", "P90", "predicted_revenue"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    grouped = (
        df.groupby(["horizon_days", "forecast_month"], as_index=False)
        .agg(
            {
                "P10": "sum",
                "P50": "sum",
                "P90": "sum",
                "predicted_revenue": "sum",
            }
        )
        .sort_values(["horizon_days", "forecast_month"])
    )

    records: List[dict] = []

    for _, row in grouped.iterrows():
        records.append(
            {
                "payer_code": "ALL",
                "pack_type": "ALL",
                "forecast_month": str(row["forecast_month"] or ""),
                "horizon_days": int(row["horizon_days"]),
                "predicted_revenue": clean_numeric(row["predicted_revenue"]),
                "P10": clean_numeric(row["P10"]),
                "P50": clean_numeric(row["P50"]),
                "P90": clean_numeric(row["P90"]),
            }
        )

    return records