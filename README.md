# 🌧️ Aplikasi Prediksi Curah Hujan - Kabupaten Bogor

Aplikasi web berbasis **Streamlit** untuk memprediksi curah hujan di Kabupaten Bogor menggunakan model **LSTM** dan **Prophet**.

## 📋 Fitur Utama

### 🏠 Home
- Informasi overview aplikasi
- Deskripsi dataset dan fitur
- Panduan penggunaan aplikasi

### 📊 Data
- Eksplorasi dataset BMKG
- Visualisasi time series (Suhu, Kelembapan, Curah Hujan)
- Statistik deskriptif
- Analisis korelasi fitur
- Distribusi data

### 📈 Evaluasi Model
- Performa model LSTM
- Performa model Prophet
- Perbandingan metrik (MAE, MSE, RMSE)
- Visualisasi hasil evaluasi

### 🔮 Prediksi
- Input manual untuk suhu dan kelembapan
- Prediksi menggunakan LSTM
- Prediksi menggunakan Prophet
- Kategori intensitas hujan
- Interpretasi hasil prediksi

## 🚀 Cara Menjalankan Aplikasi

### 1. Instalasi Dependencies

```bash
pip install -r requirements.txt
```

### 2. Struktur Folder

Pastikan struktur folder Anda seperti berikut:

```
PROJECT_BUHELA/
├── app/
│   ├── streamlit_app.py      # Aplikasi Streamlit utama
│   ├── requirements.txt       # Dependencies
│   ├── data/
│   │   └── data_bmkg_raw.csv # Dataset
│   ├── model/
│   │   ├── lstm_model_rr.keras      # Model LSTM
│   │   └── prophet_model_rr.joblib  # Model Prophet
│   └── scaler/
│       ├── scaler_features.joblib   # Scaler untuk fitur
│       └── scaler_target.joblib     # Scaler untuk target
```

### 3. Jalankan Aplikasi

```bash
streamlit run streamlit_app.py
```

Aplikasi akan terbuka di browser Anda di alamat: `http://localhost:8501`

## 📊 Dataset

Dataset yang digunakan berasal dari **BMKG (Badan Meteorologi, Klimatologi, dan Geofisika)** dengan periode data dari tahun 2010 hingga 2025.

### Fitur Dataset:
- **date**: Tanggal pengamatan
- **TAVG**: Suhu rata-rata (°C)
- **RH_AVG**: Kelembapan rata-rata (%)
- **RR**: Curah hujan (mm) - **Target Prediksi**

## 🤖 Model Machine Learning

### 1. LSTM (Long Short-Term Memory)
- Model deep learning untuk time series
- Arsitektur: 4 layers LSTM dengan Dropout
- Input: 7 hari data historis (look_back=7)
- Output: Prediksi curah hujan 1 hari ke depan

### 2. Prophet
- Model forecasting dari Facebook
- Menangkap seasonality dan trend
- Menggunakan regressors: TAVG dan RH_AVG

## 📈 Metrik Evaluasi

- **MAE (Mean Absolute Error)**: Error rata-rata absolut
- **MSE (Mean Squared Error)**: Error kuadrat rata-rata
- **RMSE (Root Mean Squared Error)**: Akar dari MSE

## 🎨 Teknologi yang Digunakan

- **Streamlit**: Framework untuk web application
- **TensorFlow/Keras**: Deep learning framework untuk LSTM
- **Prophet**: Time series forecasting
- **Pandas**: Data manipulation
- **Matplotlib & Seaborn**: Data visualization
- **Scikit-learn**: Preprocessing dan evaluasi

## 📍 Lokasi

**Kabupaten Bogor, Jawa Barat, Indonesia**

## 🔧 Troubleshooting

### Error: Model tidak ditemukan
Pastikan semua file model ada di direktori yang benar:
- `model/lstm_model_rr.keras`
- `model/prophet_model_rr.joblib`
- `scaler/scaler_features.joblib`
- `scaler/scaler_target.joblib`

### Error: Data tidak ditemukan
Pastikan file dataset ada di:
- `data/data_bmkg_raw.csv`

### Error: Import module
Pastikan semua dependencies sudah terinstall:
```bash
pip install -r requirements.txt
```

## 📝 Catatan

- Aplikasi ini dibuat untuk tujuan edukasi dan penelitian
- Prediksi curah hujan bersifat estimasi dan dapat berbeda dengan kondisi aktual
- Untuk prediksi yang lebih akurat, gunakan data real-time terbaru

## 👨‍💻 Developer

Dikembangkan sebagai bagian dari proyek Deep Learning - Prediksi Curah Hujan

---

© 2025 - Sistem Prediksi Curah Hujan Kabupaten Bogor
