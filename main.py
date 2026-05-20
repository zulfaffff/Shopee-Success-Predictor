import os
import sys
import joblib
import pandas as pd
import numpy as np
import sqlite3  # Bawaan Python, tidak perlu pip install
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Daftarkan folder src agar modul preprocess bisa di-import
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.preprocess import run_full_preprocessing

# 1. Inisialisasi Aplikasi FastAPI
app = FastAPI(
    title="Shopee Success Predictor API",
    description="REST API to predict whether a Shopee phone case product will be successful (>= 1000 units sold) or not.",
    version="1.0"
)

# 2. Inisialisasi Database SQLite & Buat Tabel Logging jika belum ada
DB_PATH = "prediction_logs.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Buat tabel untuk mencatat semua history input dan hasil prediksi
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            product_title TEXT,
            current_price TEXT,
            prediction_gb TEXT,
            probability_gb REAL,
            prediction_rf TEXT
        )
    """)
    conn.commit()
    conn.close()

# Jalankan fungsi database saat API pertama kali dinyalakan
init_db()

# 3. Load Model dan Vectorizer
try:
    model_gb = joblib.load('models/shopee_model_gb.pkl')
    model_rf = joblib.load('models/shopee_model_rf.pkl')
    tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    print("Models and Vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading models: {str(e)}")

# 4. Definisikan Skema Data Input Menggunakan Pydantic
class ProductInput(BaseModel):
    product_title: str
    original_price: str
    current_price: str
    units_sold: str
    rating_score: str
    rating_count: str
    desc: str
    img_count: int

# 5. Endpoint Root
@app.get("/")
def home():
    return {"message": "Shopee Success Predictor API is Running!"}

# 6. Endpoint Prediksi (POST Method) dengan Fitur Auto-Logging
@app.post("/predict")
def predict_product_success(payload: ProductInput):
    try:
        # a. Ubah payload JSON dari user menjadi Pandas DataFrame
        raw_data = {
            'product_title': [payload.product_title],
            'original_price': [payload.original_price],
            'current_price': [payload.current_price],
            'units_sold': [payload.units_sold],
            'rating_score': [payload.rating_score],
            'rating_count': [payload.rating_count],
            'desc': [payload.desc],
            'img_count': [payload.img_count]
        }
        df_input = pd.DataFrame(raw_data)

        # b. Jalankan fungsi preprocessing otomatis
        df_cleaned = run_full_preprocessing(df_input)

        # c. Ekstrak fitur tekstual menggunakan TF-IDF
        text_features_vals = tfidf_vectorizer.transform(df_cleaned['text_final']).toarray()

        # d. Ekstrak fitur numerik
        numerical_features = ['current_price_clean', 'discount_pct', 'name_len', 'desc_len', 'img_count']
        num_features_vals = df_cleaned[numerical_features].values

        # e. Gabungkan fitur numerik dan tekstual
        X_combined = np.hstack((num_features_vals, text_features_vals))

        # f. Lakukan prediksi
        prediction_gb = int(model_gb.predict(X_combined)[0])
        prob_gb = float(model_gb.predict_proba(X_combined)[0][1])
        prediction_rf = int(model_rf.predict(X_combined)[0])

        label_gb = "Successful (Laris)" if prediction_gb == 1 else "Unsuccessful (Tidak Laris)"
        label_rf = "Successful (Laris)" if prediction_rf == 1 else "Unsuccessful (Tidak Laris)"

        # ================= INI FITUR LOGGING DATABASE BARU =================
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO logs (timestamp, product_title, current_price, prediction_gb, probability_gb, prediction_rf)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                payload.product_title,
                payload.current_price,
                label_gb,
                round(prob_gb * 100, 2),
                label_rf
            ))
            conn.commit()
            conn.close()
            print("LOGGED TO DATABASE: Success recorded!")
        except Exception as db_err:
            print(f"Database logging failed: {str(db_err)}")
        # ===================================================================

        # Return response JSON ke user
        return {
            "status": "success",
            "prediction_gradient_boosting": {
                "label": label_gb,
                "success_probability": round(prob_gb * 100, 2)
            },
            "prediction_random_forest": {
                "label": label_rf
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")