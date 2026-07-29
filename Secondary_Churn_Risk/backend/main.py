from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import sys

# Import model pipeline
from model import run_pipeline

app = FastAPI(title="Secondary Data Churn Risk API")

# Setup CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174", "http://localhost:5173", "http://127.0.0.1:5174", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@app.get("/api/secondary-predictions")
def get_predictions():
    # Run the pipeline (which reads the local parquet)
    try:
        results = run_pipeline(BASE_DIR)
        return {"data": results, "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "failed"}

if __name__ == "__main__":
    import uvicorn
    # Critical: Serve on Port 8001
    uvicorn.run(app, host="0.0.0.0", port=8001)
