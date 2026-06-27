# 📊 Forecasting Penjualan - XGBoost & LightGBM

Aplikasi web **Streamlit** untuk forecasting (peramalan) penjualan 4 kategori produk menggunakan model **XGBoost** dan **LightGBM**.

> 🎓 Proyek Skripsi

## 🚀 Fitur

- Forecasting penjualan 1 bulan ke depan untuk 4 kategori produk
- Perbandingan model XGBoost vs LightGBM
- Visualisasi interaktif hasil prediksi
- Upload data Excel untuk prediksi custom
- Download hasil forecast (CSV/Excel)
- Evaluasi performa model (RMSE, MAE, MAPE, R²)

## 📁 Struktur Project

```
├── app.py                  # Aplikasi Streamlit utama
├── models/                 # Model ML (.joblib)
│   ├── xgboost_4kategori.joblib
│   └── lightgbm_4kategori.joblib
├── Data/                   # Dataset
│   ├── LPJ_KOPKAR_2024_2025 Baru.xlsx
│   └── Hari_Libur_2024_2025.xlsx
├── outputs/                # Output training
│   └── model_4kategori/
├── .streamlit/config.toml  # Konfigurasi Streamlit
├── requirements.txt        # Dependencies Python
└── packages.txt            # System packages
```

## 🛠️ Tech Stack

- **Python 3.12**
- **Streamlit** - Web framework
- **XGBoost** - Gradient boosting model
- **LightGBM** - Light gradient boosting model
- **Pandas & NumPy** - Data processing
- **Plotly** - Interactive visualization

## 💻 Jalankan Lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📝 Lisensi

Proyek ini dibuat untuk keperluan akademik (Skripsi).
