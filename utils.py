import pandas as pd
import numpy as np
import joblib
import os
from dotenv import load_dotenv
import gc

# Configure TensorFlow untuk memory efficiency SEBELUM import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import tensorflow as tf
# Limit TensorFlow memory usage
try:
    # Set memory growth untuk GPU jika ada
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    # Limit CPU threads untuk hemat memory
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
except Exception as e:
    print(f"TensorFlow config warning: {e}")

from tensorflow.keras.models import load_model
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Google Gemini AI
try:
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
        gemini_available = True
        print("Gemini AI configured successfully.")
    else:
        gemini_available = False
        print("GOOGLE_API_KEY not found in environment variables.")
except Exception as e:
    gemini_available = False
    print(f"Gemini AI tidak tersedia: {e}")

# Cache untuk model dan data (di-load sekali saja)
_cached_lstm_model = None
_cached_prophet_model = None
_cached_scaler_features = None
_cached_scaler_target = None
_cached_data = None

def load_data():
    """Load and preprocess the dataset with caching."""
    global _cached_data
    
    if _cached_data is not None:
        return _cached_data
    
    try:
        # Load hanya kolom yang diperlukan untuk menghemat memory
        df = pd.read_csv('data/data_bmkg_raw.csv', usecols=['date', 'TAVG', 'RH_AVG', 'RR'])
        df['date'] = pd.to_datetime(df['date'])
        _cached_data = df
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def get_gemini_status():
    return gemini_available

def generate_ai_explanation(rainfall, tavg, rh_avg, category, model_name):
    """Generate AI-powered explanation using Google Gemini for single model"""
    if not gemini_available:
        print("Gemini not available for explanation.")
        return None
    
    try:
        # Note: Ensure 'gemma-3-4b-it' is available in your API tier. 
        # Fallback to 'gemini-1.5-flash' if needed.
        try:
            model = genai.GenerativeModel('gemma-3-4b-it')
        except:
            print("Model gemma-3-4b-it not found, falling back to gemini-1.5-flash")
            model = genai.GenerativeModel('gemini-1.5-flash')
        
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

def generate_comparison_explanation(lstm_result, prophet_result, tavg, rh_avg):
    """Generate AI-powered comparison explanation for both models"""
    if not gemini_available:
        print("Gemini not available for comparison.")
        return None
    
    try:
        try:
            model = genai.GenerativeModel('gemma-3-4b-it')
        except:
            print("Model gemma-3-4b-it not found, falling back to gemini-1.5-flash")
            model = genai.GenerativeModel('gemini-1.5-flash')
        
        lstm_rainfall = float(lstm_result['prediction'])
        prophet_rainfall = float(prophet_result['prediction'])
        diff = abs(lstm_rainfall - prophet_rainfall)
        avg_rainfall = (lstm_rainfall + prophet_rainfall) / 2
        
        prompt = f"""
        Anda adalah ahli meteorologi yang membandingkan hasil prediksi dari dua model curah hujan.
        
        Data Input:
        - Lokasi: Stasiun Meteorologi Citeko, Kabupaten Bogor
        - Suhu Rata-rata: {tavg}°C
        - Kelembapan Rata-rata: {rh_avg}%
        
        Hasil Prediksi:
        1. Model LSTM (Deep Learning):
           - Prediksi: {lstm_rainfall:.2f} mm/hari
           - Kategori: {lstm_result['category']}
        
        2. Model Prophet (Time Series):
           - Prediksi: {prophet_rainfall:.2f} mm/hari
           - Kategori: {prophet_result['category']}
        
        Perbedaan: {diff:.2f} mm/hari
        Rata-rata: {avg_rainfall:.2f} mm/hari
        
        Berikan analisis yang mencakup:
        1. **Interpretasi Hasil LSTM**: Apa yang dikatakan model LSTM tentang curah hujan?
        2. **Interpretasi Hasil Prophet**: Apa yang dikatakan model Prophet tentang curah hujan?
        3. **Perbandingan Kedua Model**: Mengapa ada perbedaan? Model mana yang lebih reliable dalam kondisi ini?
        4. **Kesimpulan & Rekomendasi**: Prediksi mana yang sebaiknya digunakan? Saran praktis untuk masyarakat.
        5. **Hubungan dengan Kondisi Cuaca**: Bagaimana suhu dan kelembapan mempengaruhi hasil prediksi.
        
        Format jawaban:
        - Gunakan subjudul untuk setiap bagian (misal: **Hasil LSTM**, **Hasil Prophet**, dll)
        - Bahasa Indonesia yang jelas dan mudah dipahami
        - Maksimal 300 kata
        - Berikan insight yang actionable
        """
        
        response = model.generate_content(prompt)
        return response.text
    
    except Exception as e:
        print(f"Error generating comparison explanation: {e}")
        return None

def predict_lstm(tavg, rh_avg):
    global _cached_lstm_model, _cached_scaler_features, _cached_scaler_target
    
    try:
        # Load model dan scaler hanya sekali
        if _cached_lstm_model is None:
            _cached_lstm_model = load_model('model/lstm_model_rr.keras')
            _cached_scaler_features = joblib.load('scaler/scaler_features.joblib')
            _cached_scaler_target = joblib.load('scaler/scaler_target.joblib')
        
        # Use DataFrame to preserve feature names (fix sklearn warning)
        input_df = pd.DataFrame([[tavg, rh_avg]], columns=['TAVG', 'RH_AVG'])
        scaled_input = _cached_scaler_features.transform(input_df)
        
        # Create sequence (repeat input 7 times as in original app)
        lstm_input = np.repeat(scaled_input, 7, axis=0).reshape(1, 7, 2)
        
        prediction_scaled = _cached_lstm_model.predict(lstm_input, verbose=0)
        
        # Use DataFrame for target scaler too
        prediction_df = pd.DataFrame(prediction_scaled, columns=['RR'])
        prediction_lstm = _cached_scaler_target.inverse_transform(prediction_df)
        prediction_lstm = np.expm1(prediction_lstm)  # Inverse log transformation
        
        return float(prediction_lstm[0][0])
    except Exception as e:
        print(f"Error in LSTM prediction: {e}")
        return None

def predict_prophet(tavg, rh_avg):
    global _cached_prophet_model
    
    try:
        # Load model hanya sekali
        if _cached_prophet_model is None:
            _cached_prophet_model = joblib.load('model/prophet_model_rr.joblib')
        
        future_df = pd.DataFrame({
            'ds': [pd.Timestamp.now()],
            'TAVG': [tavg],
            'RH_AVG': [rh_avg]
        })
        
        forecast = _cached_prophet_model.predict(future_df)
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
