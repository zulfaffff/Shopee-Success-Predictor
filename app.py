import streamlit as st
import requests

# 1. Konfigurasi Halaman Web
st.set_page_config(
    page_title="Shopee Success Predictor",
    page_icon="🛍️",
    layout="centered"
)

# URL tempat FastAPI kamu berjalan (Localhost)
API_URL = "http://127.0.0.1:8000/predict"

# 2. Desain Header / Judul Web
st.title("🛍️ Shopee Product Success Predictor")
st.markdown("""
Predict whether your mobile phone case product will reach **Successful Sales (>= 1,000 units sold)** on Shopee.  
*Powered by Gradient Boosting & Random Forest Models.*
""")
st.markdown("---")

# 3. Membuat Form Input untuk User
st.subheader("📋 Enter Product Details")

with st.form("prediction_form"):
    product_title = st.text_input("Product Title", placeholder="e.g., Softcase iPhone 13 Pro Max Silicon Matte")
    
    col1, col2 = st.columns(2)
    with col1:
        original_price = st.text_input("Original Price (with Rp)", value="Rp 50.000")
        current_price = st.text_input("Current Price (with Rp)", value="Rp 25.000")
        img_count = st.number_input("Image Count", min_value=1, max_value=9, value=5)
    
    with col2:
        units_sold = st.text_input("Units Sold Display (e.g., 1,2RB+ / 500+)", value="100+")
        rating_score = st.text_input("Rating Score (e.g., 4.8)", value="4.8")
        rating_count = st.text_input("Rating Count (e.g., 150)", value="50")
        
    desc = st.text_area("Product Description", placeholder="Write your product description here...")
    
    # Tombol Submit di dalam form
    submit_button = st.form_submit_button(label="⚡ Analyze Product Success")

# 4. Aksi Ketika Tombol Klik
if submit_button:
    # Validasi input sederhana
    if not product_title or not desc:
        st.error("Please fill in the Product Title and Description!")
    else:
        # Satukan data menjadi format JSON untuk dikirim ke FastAPI
        payload = {
            "product_title": product_title,
            "original_price": original_price,
            "current_price": current_price,
            "units_sold": units_sold,
            "rating_score": rating_score,
            "rating_count": rating_count,
            "desc": desc,
            "img_count": int(img_count)
        }
        
        with st.spinner("Analyzing data with Machine Learning models..."):
            try:
                # KIRIM REQUEST KE FASTAPI (main.py)
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    st.success("Analysis Completed!")
                    st.markdown("### 📊 Prediction Results")
                    
                    # Tampilkan Hasil Gradient Boosting (Model Utama)
                    gb_res = result["prediction_gradient_boosting"]
                    gb_label = gb_res["label"]
                    gb_prob = gb_res["success_probability"]
                    
                    # Beri warna berdasarkan hasil prediksi
                    if "Successful" in gb_label:
                        st.metric(label="Gradient Boosting Prediction", value=gb_label, delta=f"{gb_prob}% Confidence")
                        st.balloons() # Efek balon kalau laris!
                    else:
                        st.metric(label="Gradient Boosting Prediction", value=gb_label, delta=f"{gb_prob}% Confidence", delta_color="inverse")
                    
                    # Tampilkan Hasil Random Forest (Model Pembanding)
                    rf_label = result["prediction_random_forest"]["label"]
                    st.info(f"**Random Forest Benchmark:** {rf_label}")
                    
                else:
                    st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the Backend API (main.py). Is your Uvicorn server running?")