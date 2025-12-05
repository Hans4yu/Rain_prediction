import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Gemini AI
try:
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_available = True
    else:
        gemini_available = False
except Exception as e:
    gemini_available = False
    print(f"Gemini AI tidak tersedia: {e}")

def load_data():
    """Load and preprocess the dataset."""
    try:
        df = pd.read_csv('data/data_bmkg_raw.csv')
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def get_gemini_status():
    return gemini_available

def generate_ai_explanation(rainfall, tavg, rh_avg, category, model_name):
    """Generate AI-powered explanation using Google Gemini"""
    if not gemini_available:
        return None
    
    try:
        model = genai.GenerativeModel('gemma-3-4b-it')
        
        prompt = f"""
        Anda adalah ahli meteorologi yang memberikan interpretasi prediksi curah hujan.
        
        Data Prediksi:
        - Lokasi: Stasiun Meteorologi Citeko, Kabupaten Bogor
        - Model: {model_name}
        - Curah Hujan Prediksi: {rainfall:.2f} mm/hari
        - Kategori: {category}
        - Suhu Rata-rata: {tavg}°C
        - Kelembapan Rata-rata: {rh_avg}%
        
        Berikan penjelasan yang:
        1. Menginterpretasikan hasil prediksi curah hujan
        2. Menjelaskan hubungan antara suhu, kelembapan, dan curah hujan
        3. Memberikan saran praktis untuk masyarakat (misalnya: persiapan menghadapi hujan, aktivitas yang sesuai)
        4. Dampak potensial dari intensitas hujan tersebut
        5. Gunakan bahasa Indonesia yang mudah dipahami
        
        Jawaban harus dalam format paragraf yang informatif dan praktis (maksimal 200 kata).
        """
        
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        print(f"Error generating AI explanation: {e}")
        return None

def predict_lstm(tavg, rh_avg):
    try:
        model_lstm = load_model('model/lstm_model_rr.keras')
        scaler_features = joblib.load('scaler/scaler_features.joblib')
        scaler_target = joblib.load('scaler/scaler_target.joblib')
        
        input_features = np.array([[tavg, rh_avg]])
        scaled_input = scaler_features.transform(input_features)
        
        # Create sequence (repeat input 7 times as in original app)
        lstm_input = np.repeat(scaled_input, 7, axis=0).reshape(1, 7, 2)
        
        prediction_scaled = model_lstm.predict(lstm_input, verbose=0)
        prediction_lstm = scaler_target.inverse_transform(prediction_scaled)
        prediction_lstm = np.expm1(prediction_lstm)  # Inverse log transformation
        
        return float(prediction_lstm[0][0])
    except Exception as e:
        print(f"Error in LSTM prediction: {e}")
        return None

def predict_prophet(tavg, rh_avg):
    try:
        model_prophet = joblib.load('model/prophet_model_rr.joblib')
        
        future_df = pd.DataFrame({
            'ds': [pd.Timestamp.now()],
            'TAVG': [tavg],
            'RH_AVG': [rh_avg]
        })
        
        forecast = model_prophet.predict(future_df)
        prediction_prophet = np.expm1(forecast['yhat'].values[0])
        
        return float(prediction_prophet)
    except Exception as e:
        print(f"Error in Prophet prediction: {e}")
        return None

def get_rain_category(rainfall):
    if rainfall < 0.5:
        return "Tidak Ada Hujan ☀️", "text-blue-400", "< 0.5 mm/hari"
    elif rainfall < 20:
        return "Hujan Ringan 🌤️", "text-green-500", "0.5 - 20 mm/hari"
    elif rainfall < 50:
        return "Hujan Sedang 🌦️", "text-yellow-500", "20 - 50 mm/hari"
    elif rainfall < 100:
        return "Hujan Lebat 🌧️", "text-red-500", "50 - 100 mm/hari"
    elif rainfall < 150:
        return "Hujan Sangat Lebat ⛈️", "text-purple-600", "100 - 150 mm/hari"
    else:
        return "Hujan Ekstrem 🌊", "text-red-700", "> 150 mm/hari"
