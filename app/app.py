import os
import requests
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Fraud Detection",
    page_icon="🛡️",
    layout="wide"
)


# ============================================================
# FASTAPI CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000/predict"


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "xgb_behavior.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    BASE_DIR,
    "models",
    "preprocessor_b.pkl"
)

FEATURE_NAMES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "feature_names_b.pkl"
)


# ============================================================
# LOAD MODEL ONLY FOR SHAP
# ============================================================

model = joblib.load(MODEL_PATH)

preprocessor = joblib.load(
    PREPROCESSOR_PATH
)

feature_names = joblib.load(
    FEATURE_NAMES_PATH
)


# ============================================================
# SHAP EXPLAINER
# ============================================================

explainer = shap.TreeExplainer(model)


# ============================================================
# FEATURE NAME CLEANER
# ============================================================

def clean_feature_name(feature):

    # Remove preprocessing prefix
    feature = feature.replace(
        "remainder__",
        ""
    )

    # Grouped transaction type
    if feature == "transaction_type":
        return "Transaction Type"

    # Original transaction type features
    if feature.startswith("cat__type_"):

        transaction_type = feature.replace(
            "cat__type_",
            ""
        )

        return "Transaction Type = " + transaction_type

    # Friendly feature names
    feature_mapping = {

        "hour":
            "Transaction Hour",

        "step":
            "Transaction Step",

        "amount":
            "Transaction Amount",

        "oldbalanceOrg":
            "Sender Old Balance",

        "newbalanceOrig":
            "Sender New Balance",

        "oldbalanceDest":
            "Receiver Old Balance",

        "newbalanceDest":
            "Receiver New Balance",

        "prev_amount":
            "Previous Transaction Amount",

        "time_since_prev":
            "Time Since Previous Transaction",

        "prior_transaction_count":
            "Previous Transaction Count",

        "prior_avg_amount":
            "Previous Average Transaction Amount",

        "amount_vs_prior_avg":
            "Amount vs Previous Average",

        "has_previous_transaction":
            "Has Previous Transaction",

        "isFlaggedFraud":
            "Existing Fraud Flag"
    }

    return feature_mapping.get(
        feature,
        feature
    )


# ============================================================
# TITLE
# ============================================================

st.title(
    "🛡️ AI-Powered Fraud Detection System"
)

st.write(
    "Interactive transaction fraud detection, "
    "risk scoring and Explainable AI"
)

st.success(
    "Connected to FastAPI prediction service"
)


# ============================================================
# TRANSACTION DETAILS
# ============================================================

st.header(
    "💳 Transaction Details"
)


# ============================================================
# INPUT COLUMNS
# ============================================================

col1, col2 = st.columns(2)


# ============================================================
# LEFT COLUMN
# ============================================================

with col1:

    transaction_type = st.selectbox(
        "Transaction Type",
        [
            "CASH_IN",
            "CASH_OUT",
            "DEBIT",
            "PAYMENT",
            "TRANSFER"
        ]
    )

    amount = st.number_input(
        "Transaction Amount",
        min_value=0.0,
        value=10000.0
    )

    old_balance_org = st.number_input(
        "Sender's Old Balance",
        min_value=0.0,
        value=10000.0
    )

    old_balance_dest = st.number_input(
        "Receiver's Old Balance",
        min_value=0.0,
        value=0.0
    )

    is_flagged_fraud = st.selectbox(
        "Existing Fraud Flag",
        [0, 1]
    )


# ============================================================
# RIGHT COLUMN
# ============================================================

with col2:

    step = st.number_input(
        "Transaction Step",
        min_value=0,
        value=100
    )

    hour = st.number_input(
        "Transaction Hour",
        min_value=0,
        max_value=23,
        value=12
    )

    prev_amount = st.number_input(
        "Previous Transaction Amount",
        min_value=0.0,
        value=0.0
    )

    time_since_prev = st.number_input(
        "Time Since Previous Transaction",
        min_value=0.0,
        value=0.0
    )

    prior_transaction_count = st.number_input(
        "Previous Transaction Count",
        min_value=0,
        value=0
    )

    prior_avg_amount = st.number_input(
        "Average Previous Transaction Amount",
        min_value=0.0,
        value=0.0
    )


# ============================================================
# BEHAVIORAL FEATURES
# ============================================================

amount_vs_prior_avg = st.number_input(
    "Amount vs Previous Average",
    min_value=0.0,
    value=1.0
)

has_previous_transaction = st.selectbox(
    "Has Previous Transaction?",
    [0, 1]
)


# ============================================================
# PREDICT FRAUD
# ============================================================

if st.button(
    "🔍 Predict Fraud",
    use_container_width=True
):

    # ========================================================
    # CREATE TRANSACTION DATA
    # ========================================================

    transaction_data = {

        "step": step,

        "type": transaction_type,

        "amount": amount,

        "oldbalanceOrg":
            old_balance_org,

        "oldbalanceDest":
            old_balance_dest,

        "isFlaggedFraud":
            is_flagged_fraud,

        "hour": hour,

        "prev_amount":
            prev_amount,

        "time_since_prev":
            time_since_prev,

        "prior_transaction_count":
            prior_transaction_count,

        "prior_avg_amount":
            prior_avg_amount,

        "amount_vs_prior_avg":
            amount_vs_prior_avg,

        "has_previous_transaction":
            has_previous_transaction
    }


    # ========================================================
    # SEND TRANSACTION TO FASTAPI
    # ========================================================

    try:

        with st.spinner(
            "Sending transaction to fraud detection API..."
        ):

            response = requests.post(
                API_URL,
                json=transaction_data,
                timeout=30
            )


        # ====================================================
        # CHECK API RESPONSE
        # ====================================================

        if response.status_code != 200:

            st.error(
                f"❌ FastAPI returned error "
                f"{response.status_code}"
            )

            st.code(
                response.text
            )

            st.stop()


        # ====================================================
        # GET JSON RESPONSE
        # ====================================================

        result = response.json()


        # ====================================================
        # GET PREDICTION RESULTS
        # ====================================================

        fraud_probability = float(
            result["fraud_probability"]
        )

        prediction = int(
            result["prediction"]
        )

        risk_level = result[
            "risk_level"
        ]

        decision = result[
            "decision"
        ]

        threshold = float(
            result["threshold"]
        )


        # ====================================================
        # RISK SCORE
        # ====================================================

        risk_score = round(
            fraud_probability * 100,
            2
        )


        # ====================================================
        # FRAUD RISK ASSESSMENT
        # ====================================================

        st.header(
            "📊 Fraud Risk Assessment"
        )

        col1, col2, col3, col4 = (
            st.columns(4)
        )


        col1.metric(
            "Fraud Probability",
            f"{fraud_probability * 100:.2f}%"
        )

        col2.metric(
            "Risk Score",
            f"{risk_score:.2f}/100"
        )

        col3.metric(
            "Risk Level",
            risk_level
        )

        col4.metric(
            "Decision",
            decision
        )


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        if risk_level == "HIGH":

            st.error(
                "🚨 Model classified this transaction "
                "as potentially fraudulent."
            )

        elif risk_level == "MEDIUM":

            st.warning(
                "⚠️ Model identified this transaction "
                "as requiring further review."
            )

        else:

            st.success(
                "✅ Model classified this transaction "
                "as low risk."
            )


        # ====================================================
        # RISK INTERPRETATION
        # ====================================================

        if risk_level == "HIGH":

            st.error(
                "🚨 High-risk transaction. "
                "The transaction should be blocked "
                "or sent for fraud investigation."
            )

        elif risk_level == "MEDIUM":

            st.warning(
                "⚠️ Medium-risk transaction. "
                "Manual review is recommended."
            )

        else:

            st.success(
                "✅ Low-risk transaction. "
                "The transaction can be approved."
            )


        # ====================================================
        # SHAP EXPLAINABLE AI
        # ====================================================

        st.header(
            "🧠 Explainable AI"
        )

        st.write(
            "The following factors explain why the "
            "model considered this transaction risky "
            "or legitimate."
        )


        # ====================================================
        # CREATE DATAFRAME FOR SHAP
        # ====================================================

        transaction_df = pd.DataFrame(
            [transaction_data]
        )


        # ====================================================
        # PREPROCESS FOR SHAP
        # ====================================================

        transaction_encoded = (
            preprocessor.transform(
                transaction_df
            )
        )


        # ====================================================
        # CALCULATE SHAP VALUES
        # ====================================================

        shap_values = explainer.shap_values(
            transaction_encoded
        )

        shap_values = np.asarray(
            shap_values
        )


        # ====================================================
        # HANDLE POSSIBLE SHAP OUTPUT
        # ====================================================

        if shap_values.ndim == 3:

            shap_values = shap_values[0]

        if shap_values.ndim == 2:

            shap_values = shap_values[0]

        shap_values = (
            shap_values.reshape(-1)
        )


        # ====================================================
        # CREATE ORIGINAL SHAP DATAFRAME
        # ====================================================

        explanation = pd.DataFrame({

            "feature":
                feature_names,

            "shap_value":
                shap_values
        })


        # ====================================================
        # GROUP TRANSACTION TYPE FEATURES
        # ====================================================

        grouped_explanation = []

        transaction_type_shap = 0.0


        for _, row in explanation.iterrows():

            feature = row[
                "feature"
            ]

            shap_value = row[
                "shap_value"
            ]


            # ----------------------------------------------
            # GROUP ALL ONE-HOT TRANSACTION TYPE FEATURES
            # ----------------------------------------------

            if feature.startswith(
                "cat__type_"
            ):

                transaction_type_shap += (
                    shap_value
                )

            else:

                grouped_explanation.append({

                    "feature":
                        feature,

                    "shap_value":
                        shap_value
                })


        # ====================================================
        # ADD GROUPED TRANSACTION TYPE
        # ====================================================

        grouped_explanation.append({

            "feature":
                "transaction_type",

            "shap_value":
                transaction_type_shap
        })


        # ====================================================
        # CREATE GROUPED DATAFRAME
        # ====================================================

        explanation = pd.DataFrame(
            grouped_explanation
        )


        # ====================================================
        # CALCULATE IMPACT
        # ====================================================

        explanation["impact"] = (
            explanation[
                "shap_value"
            ].abs()
        )


        # ====================================================
        # SORT FEATURES
        # ====================================================

        explanation = (
            explanation
            .sort_values(
                "impact",
                ascending=False
            )
        )


        # ====================================================
        # POSITIVE SHAP FACTORS
        # ====================================================

        positive_reasons = (
            explanation[
                explanation[
                    "shap_value"
                ] > 0
            ]
            .head(5)
        )


        # ====================================================
        # NEGATIVE SHAP FACTORS
        # ====================================================

        negative_reasons = (
            explanation[
                explanation[
                    "shap_value"
                ] < 0
            ]
            .head(5)
        )


        # ====================================================
        # DISPLAY SHAP
        # ====================================================

        col1, col2 = st.columns(2)


        # ====================================================
        # INCREASING FRAUD RISK
        # ====================================================

        with col1:

            st.subheader(
                "🔴 Factors Increasing Fraud Risk"
            )

            if len(
                positive_reasons
            ) > 0:

                for _, row in (
                    positive_reasons.iterrows()
                ):

                    clean_name = (
                        clean_feature_name(
                            row["feature"]
                        )
                    )

                    st.write(
                        f"**{clean_name}** "
                        f"(SHAP: "
                        f"+{row['shap_value']:.3f})"
                    )

            else:

                st.write(
                    "No significant factors "
                    "increasing fraud risk."
                )


        # ====================================================
        # REDUCING FRAUD RISK
        # ====================================================

        with col2:

            st.subheader(
                "🟢 Factors Reducing Fraud Risk"
            )

            if len(
                negative_reasons
            ) > 0:

                for _, row in (
                    negative_reasons.iterrows()
                ):

                    clean_name = (
                        clean_feature_name(
                            row["feature"]
                        )
                    )

                    st.write(
                        f"**{clean_name}** "
                        f"(SHAP: "
                        f"{row['shap_value']:.3f})"
                    )

            else:

                st.write(
                    "No significant factors "
                    "reducing fraud risk."
                )


        # ====================================================
        # DETAILED SHAP TABLE
        # ====================================================

        with st.expander(
            "📋 View Detailed SHAP Analysis"
        ):

            detailed_explanation = (
                explanation.copy()
            )

            detailed_explanation[
                "feature"
            ] = (
                detailed_explanation[
                    "feature"
                ]
                .apply(
                    clean_feature_name
                )
            )

            detailed_explanation = (
                detailed_explanation[
                    [
                        "feature",
                        "shap_value",
                        "impact"
                    ]
                ]
                .head(10)
            )

            st.dataframe(
                detailed_explanation,
                use_container_width=True
            )


        # ====================================================
        # SUCCESS MESSAGE
        # ====================================================

        st.success(
            "✅ Prediction completed successfully. "
            "FastAPI handled the prediction and "
            "saved the transaction to SQL Server."
        )


    # ========================================================
    # API CONNECTION ERROR
    # ========================================================

    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Could not connect to FastAPI."
        )

        st.info(
            "Make sure FastAPI is running with:"
        )

        st.code(
            "uvicorn api.main:app --reload"
        )


    # ========================================================
    # TIMEOUT ERROR
    # ========================================================

    except requests.exceptions.Timeout:

        st.error(
            "❌ FastAPI request timed out."
        )


    # ========================================================
    # OTHER ERRORS
    # ========================================================

    except Exception as e:

        st.error(
            "❌ Something went wrong."
        )

        st.exception(e)