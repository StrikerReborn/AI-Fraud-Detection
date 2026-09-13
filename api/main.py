from pathlib import Path

import joblib
import pandas as pd
import pyodbc

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# ============================================================
# LOAD TRAINED MODEL AND SUPPORTING FILES
# ============================================================

model = joblib.load(MODEL_DIR / "xgb_behavior.pkl")
preprocessor = joblib.load(MODEL_DIR / "preprocessor_b.pkl")
threshold = joblib.load(MODEL_DIR / "threshold.pkl")


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI Fraud Detection API",
    description="XGBoost-based fraud detection and risk assessment API",
    version="1.0.0"
)


# ============================================================
# TRANSACTION REQUEST MODEL
# ============================================================

class TransactionRequest(BaseModel):

    type: str

    step: int = Field(ge=0)

    amount: float = Field(ge=0)

    oldbalanceOrg: float = Field(ge=0)

    oldbalanceDest: float = Field(ge=0)

    isFlaggedFraud: int = Field(ge=0, le=1)

    hour: int = Field(ge=0, le=23)

    prev_amount: float = Field(ge=0)

    time_since_prev: float = Field(ge=0)

    prior_transaction_count: int = Field(ge=0)

    prior_avg_amount: float = Field(ge=0)

    amount_vs_prior_avg: float = Field(ge=0)

    has_previous_transaction: int = Field(ge=0, le=1)


# ============================================================
# HOME ENDPOINT
# ============================================================

@app.get("/")
def home():

    return {
        "message": "AI Fraud Detection API is running",
        "model": "XGBoost",
        "threshold": threshold
    }


# ============================================================
# HEALTH CHECK ENDPOINT
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }


# ============================================================
# RISK ENGINE
# ============================================================

def risk_engine(fraud_probability):

    if fraud_probability >= 0.90:
        return "HIGH", "BLOCK"

    elif fraud_probability >= 0.30:
        return "MEDIUM", "REVIEW"

    else:
        return "LOW", "APPROVE"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_db_connection():

    connection_string = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        r"SERVER=DESKTOP-HPTKU37\MSSQL;"
        "DATABASE=FraudDetectionDB;"
        "Trusted_Connection=yes;"
    )

    return pyodbc.connect(connection_string)


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(transaction: TransactionRequest):

    try:

        # ----------------------------------------------------
        # Convert request to DataFrame
        # ----------------------------------------------------

        data = pd.DataFrame([transaction.model_dump()])


        # ----------------------------------------------------
        # Apply preprocessing
        # ----------------------------------------------------

        processed_data = preprocessor.transform(data)


        # ----------------------------------------------------
        # Generate fraud probability
        # ----------------------------------------------------

        fraud_probability = model.predict_proba(
            processed_data
        )[0][1]


        fraud_probability = float(fraud_probability)


        # ----------------------------------------------------
        # Generate binary prediction
        # ----------------------------------------------------

        prediction = int(
            fraud_probability >= threshold
        )


        # ----------------------------------------------------
        # Apply risk engine
        # ----------------------------------------------------

        risk_level, decision = risk_engine(
            fraud_probability
        )


        # ----------------------------------------------------
        # Save prediction to SQL Server
        # ----------------------------------------------------

        connection = get_db_connection()
        cursor = connection.cursor()

        try:

            cursor.execute(
                """
                INSERT INTO FraudTransactions
                (
                    Step,
                    TransactionType,
                    Amount,
                    OldBalanceOrg,
                    OldBalanceDest,
                    IsFlaggedFraud,
                    Hour,
                    PreviousAmount,
                    TimeSincePrevious,
                    PriorTransactionCount,
                    PriorAverageAmount,
                    AmountVsPreviousAverage,
                    HasPreviousTransaction,
                    FraudProbability,
                    RiskScore,
                    RiskLevel,
                    Decision
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    transaction.step,
                    transaction.type,
                    transaction.amount,
                    transaction.oldbalanceOrg,
                    transaction.oldbalanceDest,
                    transaction.isFlaggedFraud,
                    transaction.hour,
                    transaction.prev_amount,
                    transaction.time_since_prev,
                    transaction.prior_transaction_count,
                    transaction.prior_avg_amount,
                    transaction.amount_vs_prior_avg,
                    transaction.has_previous_transaction,
                    fraud_probability,
                    fraud_probability * 100,
                    risk_level,
                    decision
                )
            )

            connection.commit()

        finally:

            cursor.close()
            connection.close()


        # ----------------------------------------------------
        # Return prediction response
        # ----------------------------------------------------

        return {
            "fraud_probability": fraud_probability,
            "prediction": prediction,
            "risk_level": risk_level,
            "decision": decision,
            "threshold": float(threshold)
        }


    except pyodbc.Error as error:

        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(error)}"
        )


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(error)}"
        )
        
        