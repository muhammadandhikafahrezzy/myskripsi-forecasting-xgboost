# Ringkasan Model Peramalan 4 Kategori

Sumber data: `C:\Skripsi\FINAL\FINAL Code - Salin\outputs\model_4kategori\dataset_harian_4kategori.csv`
Kategori: AIR MINERAL, MINUMAN TEH, ROKOK, SUSU
Rentang tanggal data harian: 2024-01-03 sampai 2025-12-22
Jumlah baris harian sebelum fitur: 1,800
Jumlah baris setelah fitur lag/rolling: 1,688

## Split Data

- Train: 2024-02-10 sampai 2025-05-12 (269 tanggal)
- Validation: 2025-05-15 sampai 2025-08-24 (68 tanggal)
- Test: 2025-08-25 sampai 2025-12-22 (85 tanggal)

## Evaluasi Test

Basis evaluasi utama: rata-rata harian per bulan dari data test. Basis ini dipakai karena tujuan aplikasi adalah forecasting periode bulanan, sementara data harian mentah memiliki fluktuasi tinggi.

| ALGORITMA | KATEGORI | BASIS_EVALUASI | N_OBSERVASI_EVALUASI | RMSE | MAPE |
| --- | --- | --- | --- | --- | --- |
| LightGBM | AIR MINERAL | Rata-rata harian per bulan | 5 | 46.7303 | 11.2449 |
| LightGBM | MINUMAN TEH | Rata-rata harian per bulan | 5 | 13.4905 | 7.8279 |
| LightGBM | ROKOK | Rata-rata harian per bulan | 5 | 4.9811 | 4.9940 |
| LightGBM | SEMUA_KATEGORI | Rata-rata harian per bulan | 20 | 24.5270 | 9.7194 |
| LightGBM | SUSU | Rata-rata harian per bulan | 5 | 3.9721 | 14.8107 |
| XGBoost | AIR MINERAL | Rata-rata harian per bulan | 5 | 50.7101 | 11.8696 |
| XGBoost | MINUMAN TEH | Rata-rata harian per bulan | 5 | 18.8595 | 8.8147 |
| XGBoost | ROKOK | Rata-rata harian per bulan | 5 | 9.1963 | 11.1296 |
| XGBoost | SEMUA_KATEGORI | Rata-rata harian per bulan | 20 | 27.5171 | 11.7547 |
| XGBoost | SUSU | Rata-rata harian per bulan | 5 | 4.1229 | 15.2049 |

Algoritma terbaik berdasarkan RMSE keseluruhan: **LightGBM**.

## Hyperparameter Terbaik

### XGBoost

```json
{
  "colsample_bytree": 0.9,
  "learning_rate": 0.05,
  "max_depth": 2,
  "min_child_weight": 1.0,
  "n_estimators": 100,
  "n_jobs": 2,
  "objective": "reg:squarederror",
  "random_state": 42,
  "reg_lambda": 1.0,
  "subsample": 0.9,
  "target_transform": "log1p",
  "tree_method": "hist"
}
```

### LightGBM

```json
{
  "colsample_bytree": 0.9,
  "force_col_wise": true,
  "learning_rate": 0.05,
  "min_child_samples": 5,
  "n_estimators": 200,
  "n_jobs": 2,
  "num_leaves": 7,
  "objective": "regression",
  "random_state": 42,
  "subsample": 0.9,
  "target_transform": "log1p",
  "verbose": -1
}
```
