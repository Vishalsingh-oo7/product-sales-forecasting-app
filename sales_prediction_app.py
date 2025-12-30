
from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

# ------------------------- #
# Flask App Initialization  #
# ------------------------- #
app = Flask(__name__)

# ------------------------- #
# Global objects (in-memory) #
# ------------------------- #
model = None
scaler = None
all_feature_cols = None

MODEL_PATH = "sales_forecasting_model.pkl"
SCALER_PATH = "scaler.pkl"
X_TRAIN_COLUMNS_PATH = "X_encoded_columns.pkl"

# These need to be consistent with how they were defined during training
cat_cols = ['Store_Type', 'Location_Type', 'Region_Code']
num_cols = ['Year', 'Month', 'Week', 'Day', 'DayOfWeek', 'Quarter']

# ------------------------------------------ #
# Feature Engineering Functions (from notebook) #
# ------------------------------------------ #
def standardize_dtypes(df_input):
    df_temp = df_input.copy(deep=True)

    df_temp.columns = df_temp.columns.str.strip()
    df_temp.rename(columns={'#Order': 'Order'}, inplace=True, errors='ignore')

    if 'Date' in df_temp.columns:
        df_temp['Date'] = pd.to_datetime(df_temp['Date'], errors='coerce')

    if 'Discount' in df_temp.columns:
        df_temp['Discount'] = (
            df_temp['Discount']
            .astype(str)
            .str.strip()
            .str.lower()
            .map({'yes': 1, 'no': 0})
        )
    return df_temp

def feature_engineering(df_input):
    df_temp = df_input.copy(deep=True)

    if 'Date' in df_temp.columns:
        df_temp['Year'] = df_temp['Date'].dt.year
        df_temp['Month'] = df_temp['Date'].dt.month
        df_temp['Week'] = df_temp['Date'].dt.isocalendar().week.astype(int)
        df_temp['Day'] = df_temp['Date'].dt.day
        df_temp['DayOfWeek'] = df_temp['Date'].dt.dayofweek
        df_temp['Is_Weekend'] = df_temp['DayOfWeek'].isin([5, 6]).astype(int)
        df_temp['Quarter'] = df_temp['Date'].dt.quarter

    df_temp.drop(columns=['ID'], inplace=True, errors='ignore')
    df_temp.drop(columns=['Store_id'], inplace=True, errors='ignore')

    return df_temp

# ---------------------------------------------------- #
# Load Model, Scaler, and Training Columns ONCE at startup #
# ---------------------------------------------------- #
def load_artifacts():
    global model, scaler, all_feature_cols

    with open(MODEL_PATH, "rb") as model_file:
        model = pickle.load(model_file)

    with open(SCALER_PATH, "rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    with open(X_TRAIN_COLUMNS_PATH, "rb") as f:
        all_feature_cols = pickle.load(f)

    print("✅ Model, Scaler, and Training Columns loaded into memory")

# ------------------------- #
# Call loader at startup    #
# ------------------------- #
load_artifacts()

# ------------------------- #
# Health Check Endpoint     #
# ------------------------- #
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "message": "Sales Forecasting API is live"
    })

# ------------------------- #
# Prediction Endpoint       #
# ------------------------- #
@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects a JSON payload with raw features.
    Example Request Body:
    {
        "Store_Type": "S1",
        "Location_Type": "L3",
        "Region_Code": "R1",
        "Date": "2019-06-01",
        "Holiday": 0,
        "Discount": "No",
        "Store_id": 171, # Optional, will be dropped
        "ID": "T1188341" # Optional, will be dropped
    }
    """
    try:
        data = request.get_json()

        # Convert incoming data to a pandas DataFrame for preprocessing
        input_df = pd.DataFrame([data])

        # Apply preprocessing functions
        input_df_processed = standardize_dtypes(input_df.copy())
        input_df_processed = feature_engineering(input_df_processed.copy())

        # Drop 'Date' column after feature engineering
        if 'Date' in input_df_processed.columns:
            input_df_processed = input_df_processed.drop(columns=['Date'])

        # One-Hot Encoding for categorical columns
        input_df_encoded = pd.get_dummies(
            input_df_processed,
            columns=cat_cols,
            drop_first=True,
            dtype=int
        )

        # Align columns with training data columns using all_feature_cols
        # This is critical to ensure the model receives features in the correct order and format
        final_features = input_df_encoded.reindex(columns=all_feature_cols, fill_value=0)

        if final_features.isnull().any().any():
            return jsonify({"error": "Missing expected features after preprocessing."}), 400

        # Scale numerical features - apply scaler only to the specified numerical columns
        final_features[num_cols] = scaler.transform(final_features[num_cols])

        # Predict
        prediction = model.predict(final_features)

        return jsonify({
            "predicted_sales": float(prediction[0])
        })

    except Exception as e:
        app.logger.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 400

# ------------------------- #
# Run Server                #
# ------------------------- #
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
