# app.py
import gradio as gr
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.cluster import KMeans

SCALER_PATH = "/tmp/scaler.pkl"
MODEL_PATH = "/tmp/rfm_classifier.pkl"

# -----------------------------
# Helper function
# -----------------------------
def clean_and_train(file):
    if file is None:
        return "❌ Please upload a CSV or XLSX file."

    file_path = file.name

    # Load data
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path, engine="openpyxl")
        else:
            return "❌ Unsupported file format."
    except Exception as e:
        return f"❌ File read error: {e}"

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # -----------------------------
    # Auto-map columns
    # -----------------------------
    date_col = None
    customer_col = None
    invoice_col = None

    for col in df.columns:
        if col in ["invoicedate", "order_date", "invoice_date"]:
            date_col = col
        if col in ["customer_id", "customerid"]:
            customer_col = col
        if col in ["invoice", "order_id", "invoice_no"]:
            invoice_col = col

    if not date_col:
        return "❌ No order date column found (InvoiceDate / OrderDate)."
    if not customer_col:
        return "❌ No customer ID column found."
    if not invoice_col:
        return "❌ No invoice/order ID column found."

    # -----------------------------
    # Data cleaning
    # -----------------------------
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    df["quantity"] = pd.to_numeric(df.get("quantity", 1), errors="coerce").fillna(1)
    df["price"] = pd.to_numeric(df.get("price", 1), errors="coerce").fillna(1)

    df["total_amount"] = df["quantity"] * df["price"]

    # -----------------------------
    # RFM calculation
    # -----------------------------
    today = df[date_col].max()

    rfm = df.groupby(customer_col).agg({
        date_col: lambda x: (today - x.max()).days,
        invoice_col: "nunique",
        "total_amount": "sum"
    })

    rfm.columns = ["recency", "frequency", "monetary"]

    # -----------------------------
    # Scaling & clustering
    # -----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm["cluster"] = kmeans.fit_predict(X_scaled)

    rfm["customer_type"] = rfm["cluster"].map({
        0: "Occasional Buyer",
        1: "Regular Shopper",
        2: "Big Spender"
    })

    # -----------------------------
    # Train classifier
    # -----------------------------
    X = rfm[["recency", "frequency", "monetary"]]
    y = rfm["customer_type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        random_state=42
    )
    clf.fit(X_train, y_train)

    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(clf, MODEL_PATH)

    y_pred = clf.predict(X_test)

    return (
        "✅ Model trained successfully!\n\n"
        f"Accuracy: {accuracy_score(y_test, y_pred):.2f}\n"
        f"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.2f}\n"
        f"Recall: {recall_score(y_test, y_pred, average='weighted', zero_division=0):.2f}"
    )

# -----------------------------
# Prediction
# -----------------------------
def predict_customer_type(recency, frequency, monetary):
    if not os.path.exists(SCALER_PATH):
        return "❌ Train the model first."

    scaler = joblib.load(SCALER_PATH)
    clf = joblib.load(MODEL_PATH)

    X = np.array([[recency, frequency, monetary]])
    X_scaled = scaler.transform(X)

    prediction = clf.predict(X_scaled)[0]

    offer_map = {
        "Occasional Buyer": "🎯 Send discount to re-engage",
        "Regular Shopper": "⭐ Offer loyalty rewards",
        "Big Spender": "💎 Invite to VIP program"
    }

    return f"Customer Type: {prediction}\nRecommended Action: {offer_map[prediction]}"

# -----------------------------
# Gradio UI
# -----------------------------
with gr.Blocks() as demo:
    gr.Markdown("## 🛍️ Customer Segmentation (RFM Model)")

    with gr.Tab("Upload & Train"):
        file_input = gr.File(label="Upload CSV / XLSX")
        train_btn = gr.Button("Clean & Train Model")
        train_output = gr.Textbox(label="Status")

        train_btn.click(clean_and_train, file_input, train_output)

    with gr.Tab("Predict"):
        r = gr.Number(label="Days since last purchase", value=30)
        f = gr.Number(label="Purchases count", value=2)
        m = gr.Number(label="Total spend", value=500)

        predict_btn = gr.Button("Predict Customer Type")
        predict_output = gr.Textbox()

        predict_btn.click(predict_customer_type, [r, f, m], predict_output)

demo.launch()
