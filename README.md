# 🌧️ Aplikasi Prediksi Curah Hujan - Kabupaten Bogor

Aplikasi web berbasis **Flask** untuk memprediksi curah hujan di Stasiun Meteorologi Citeko, Kabupaten Bogor menggunakan model **LSTM** dan **Prophet** dengan **AI-powered explanation** dari Google Gemini.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## ✨ Fitur Utama

### 🏠 Home
- Landing page dengan informasi proyek
- Overview aplikasi dan tujuan
- Navigasi ke berbagai fitur

### 📊 Data
- Eksplorasi dataset BMKG Stasiun Citeko
- Visualisasi time series interaktif (Chart.js)
- Statistik deskriptif data cuaca
- Tabel data historis (100 baris pertama)
- Downsampled chart untuk performa optimal

### 📈 Evaluasi Model
- Perbandingan performa LSTM vs Prophet
- Metrik evaluasi lengkap (MAE, MSE, RMSE)
- API endpoint untuk data evaluasi
- Visualisasi chart perbandingan

### 🔮 Prediksi
- **Input manual** suhu (TAVG) dan kelembapan (RH_AVG)
- **Prediksi LSTM** - Deep learning time series
- **Prediksi Prophet** - Facebook forecasting model
- **Kategori intensitas hujan** otomatis:
  - ☀️ Tidak Ada Hujan (< 0.5 mm)
  - 🌤️ Hujan Ringan (0.5-20 mm)
  - 🌦️ Hujan Sedang (20-50 mm)
  - 🌧️ Hujan Lebat (50-100 mm)
  - ⛈️ Hujan Sangat Lebat (100-150 mm)
  - 🌊 Hujan Ekstrem (> 150 mm)
- **AI Explanation** - Interpretasi cerdas dari Google Gemini
- **Real-time prediction** dengan AJAX

### 🤖 AI-Powered Explanation
- Menggunakan **Google Gemini AI (Gemma-3-4b-it)**
- Interpretasi hasil prediksi dalam bahasa Indonesia
- Saran praktis untuk masyarakat
- Analisis hubungan suhu, kelembapan, dan curah hujan

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/Hans4yu/Rain_prediction.git
cd Rain_prediction/app
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Setup Environment Variables

Buat file `.env` di root folder:

```bash
GOOGLE_API_KEY=your_google_gemini_api_key_here
FLASK_ENV=production
FLASK_DEBUG=False
TF_CPP_MIN_LOG_LEVEL=3
```

### 4. Struktur Folder

```
app/
├── app_flask.py              # Main Flask application
├── utils.py                  # Helper functions & model loading
├── requirements.txt          # Python dependencies
├── Procfile                  # Gunicorn configuration
├── .env.example              # Example env file
├── data/
│   └── data_bmkg_raw.csv    # BMKG dataset
├── model/
│   ├── lstm_model_rr.keras       # LSTM model
│   └── prophet_model_rr.joblib   # Prophet model
├── scaler/
│   ├── scaler_features.joblib    # Feature scaler
│   └── scaler_target.joblib      # Target scaler
└── templates/
    ├── base.html             # Base template
    ├── index.html            # Home page
    ├── data.html             # Data exploration
    ├── evaluation.html       # Model evaluation
    └── prediction.html       # Prediction interface
```

### 5. Run Locally

```bash
# Development mode
python app_flask.py

# Production mode with Gunicorn
gunicorn app_flask:app --workers 1 --threads 2 --timeout 120
```

Aplikasi akan berjalan di: `http://localhost:5000`

## 📊 Dataset

Dataset yang digunakan berasal dari **BMKG (Badan Meteorologi, Klimatologi, dan Geofisika)** dengan periode data dari tahun 2010 hingga 2025.

### Fitur Dataset:
- **date**: Tanggal pengamatan
- **TAVG**: Suhu rata-rata (°C)
- **RH_AVG**: Kelembapan rata-rata (%)
- **RR**: Curah hujan (mm) - **Target Prediksi**

## 🤖 Machine Learning Models

### 1. LSTM (Long Short-Term Memory)
- **Framework**: TensorFlow/Keras 2.13
- **Architecture**: Multi-layer LSTM dengan Dropout
- **Input**: Sequence 7 hari data (TAVG, RH_AVG)
- **Output**: Prediksi curah hujan (mm/hari)
- **Preprocessing**: 
  - Log transformation: `log1p(RR)`
  - MinMax scaling untuk features & target
  - Sequence creation dengan look_back=7
- **File**: `model/lstm_model_rr.keras` (~6 MB)

### 2. Prophet
- **Framework**: Facebook Prophet 1.1.5
- **Type**: Additive time series forecasting
- **Features**: 
  - Automatic seasonality detection
  - External regressors: TAVG, RH_AVG
  - Trend modeling
- **Preprocessing**: Log transformation `log1p(RR)`
- **File**: `model/prophet_model_rr.joblib` (~0.6 MB)

### 3. Google Gemini AI (Optional)
- **Model**: Gemma-3-4b-it
- **Purpose**: Natural language explanation
- **Language**: Indonesian
- **Features**:
  - Weather interpretation
  - Practical suggestions
  - Impact analysis

## 📈 Metrik Evaluasi

- **MAE (Mean Absolute Error)**: Error rata-rata absolut
- **MSE (Mean Squared Error)**: Error kuadrat rata-rata
- **RMSE (Root Mean Squared Error)**: Akar dari MSE

## 🎨 Tech Stack

### Backend
- **Flask 3.0** - Web framework
- **Gunicorn 21.2** - WSGI HTTP server
- **Python 3.11+** - Programming language

### Machine Learning
- **TensorFlow CPU 2.13** - LSTM model (memory-optimized)
- **Prophet 1.1.5** - Time series forecasting
- **Scikit-learn 1.3** - Preprocessing & evaluation
- **Joblib 1.3** - Model serialization

### Data Processing
- **Pandas 2.1** - Data manipulation
- **NumPy 1.24** - Numerical computing

### AI & APIs
- **Google Generative AI 0.3** - Gemini API
- **Python-dotenv 1.0** - Environment management

### Frontend
- **Tailwind CSS 3.4** - Styling framework
- **Chart.js 4.4** - Data visualization
- **Vanilla JavaScript** - Interactive features
- **Jinja2** - Template engine

### Monitoring
- **psutil 5.9** - Memory monitoring
- Custom health check endpoint

## 📍 Lokasi

**Kabupaten Bogor, Jawa Barat, Indonesia**

## 🚢 Deployment

### Render.com (Recommended)

1. Connect GitHub repository
2. Set environment variables:
   ```
   GOOGLE_API_KEY=your_key
   TF_CPP_MIN_LOG_LEVEL=3
   ```
3. Deploy automatically from `main` branch

### Railway.app

1. New Project → Deploy from GitHub
2. Add environment variables
3. Auto-detects Procfile

### Heroku

```bash
heroku create your-app-name
heroku config:set GOOGLE_API_KEY=your_key
git push heroku main
```

---

## 🔧 Troubleshooting

### ⚠️ Out of Memory Error

Aplikasi ini sudah **memory-optimized** untuk berjalan di 512MB RAM:

✅ **Optimizations Applied:**
- Model caching (load once)
- TensorFlow CPU-only version
- Garbage collection after requests
- Data downsampling
- Single worker configuration

**Solutions:**
1. Upgrade to paid tier (Render Starter $7/month)
2. See [`MEMORY_OPTIMIZATION.md`](MEMORY_OPTIMIZATION.md) for details
3. Run `python check_memory.py` to check footprint

### Error: Model tidak ditemukan

Pastikan file model ada:
```bash
ls model/
# Expected:
# lstm_model_rr.keras
# prophet_model_rr.joblib
```

### Error: GOOGLE_API_KEY not found

```bash
# Create .env file
echo "GOOGLE_API_KEY=your_api_key_here" > .env
```

Get API key: https://makersuite.google.com/app/apikey

### Error: Import module

```bash
pip install -r requirements.txt
```

### Check Application Health

```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "memory_usage_mb": "245.32",
  "models_loaded": true
}
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/data` | GET | Data exploration page |
| `/evaluation` | GET | Model evaluation page |
| `/predict` | GET/POST | Prediction interface |
| `/api/evaluation-metrics` | GET | Get evaluation metrics JSON |
| `/health` | GET | Health check & memory info |

### Example API Usage

```bash
# Health check
curl http://localhost:5000/health

# Get evaluation metrics
curl http://localhost:5000/api/evaluation-metrics

# Make prediction (POST JSON)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"tavg": 25, "rh_avg": 80, "model": "both"}'
```

---

## 📈 Performance

- **Model Loading**: ~3-5 seconds (first request)
- **Subsequent Predictions**: <1 second (cached models)
- **Memory Footprint**: ~214-400 MB (optimized)
- **Auto-restart**: Every 100 requests (prevent memory leak)

---

## 🔐 Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `GOOGLE_API_KEY` | No | Google Gemini API key | None (AI disabled) |
| `FLASK_ENV` | No | Flask environment | `production` |
| `FLASK_DEBUG` | No | Debug mode | `False` |
| `TF_CPP_MIN_LOG_LEVEL` | No | TensorFlow log level | `3` |

---

## 📝 Notes

- ✅ **Memory-optimized** untuk deployment dengan 512MB RAM
- ✅ **Production-ready** dengan Gunicorn configuration
- ✅ **AI-powered** explanations (optional, requires API key)
- ⚠️ Prediksi bersifat estimasi berdasarkan data historis
- ⚠️ Untuk akurasi terbaik, gunakan data cuaca terkini
- 📚 Built for educational & research purposes

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Developer

**Project**: Deep Learning - Rain Prediction System  
**Institution**: Semester 7 Deep Learning Course  
**Location**: Kabupaten Bogor, Indonesia  
**Year**: 2025

---

## 🙏 Acknowledgments

- **BMKG** - Data cuaca Stasiun Meteorologi Citeko
- **Google Gemini** - AI-powered explanations
- **TensorFlow Team** - Deep learning framework
- **Facebook Prophet** - Time series forecasting library

---

## 📞 Support

For issues, questions, or suggestions:
- 🐛 Open an issue on GitHub
- 📧 Contact repository owner
- 📖 Check documentation in `/docs` folder

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

[![GitHub stars](https://img.shields.io/github/stars/Hans4yu/Rain_prediction?style=social)](https://github.com/Hans4yu/Rain_prediction)
[![GitHub forks](https://img.shields.io/github/forks/Hans4yu/Rain_prediction?style=social)](https://github.com/Hans4yu/Rain_prediction/fork)

---

© 2025 - Rain Prediction System | Kabupaten Bogor

**Built with ❤️ using Flask, TensorFlow & Prophet**

</div>
