# 🏡 PREDIKSI HARGA RUMAH DI JABODETABEK MENGGUNAKAN RANDOM FOREST & XGBOOST

Pertumbuhan sektor properti di wilayah Jabodetabek semakin pesat setiap tahunnya. Namun, harga rumah di kawasan ini sangat bervariasi tergantung berbagai faktor seperti lokasi, luas tanah, jumlah kamar, kondisi bangunan, hingga status sertifikat. Ketidakteraturan harga ini menyulitkan calon pembeli maupun penjual untuk menentukan harga wajar.

Untuk menjawab permasalahan ini, proyek ini membandingkan dua model machine learning populer, Random Forest dan XGBoost, dalam melakukan prediksi harga rumah berbasis data historis properti.

---
# 🎯 TUJUAN PROYEK
  1. Mengolah dan membersihkan data harga rumah dari Jabodetabek.

  2. Melakukan eksplorasi data (EDA) dan rekayasa fitur (feature engineering).

  3. Membangun dan membandingkan performa dua model regresi:

      - Random Forest Regressor

      - XGBoost Regressor

  4. Mengevaluasi performa model menggunakan metrik:

      - MAE (Mean Absolute Error)

      - RMSE (Root Mean Squared Error)

      - R² Score
# 🔄 Alur Penyelesaian Proyek

![Data Set (1)](https://github.com/user-attachments/assets/e938047e-ff02-4a20-990f-72863b816e4c)

# 📚 Penjelasan Dataset
Dataset ini mencakup informasi rumah di wilayah Jabodetabek, dengan kombinasi fitur numerik dan kategorikal yang menggambarkan:

  - Lokasi: lat, long, district, city

  - Spesifikasi Bangunan: bedrooms, bathrooms, land_size_m2, building_size_m2, carports, floors, maid_bedrooms, dll.

  - Kondisi Bangunan: year_built, building_age, building_orientation

  - Legalitas: certificate

  - Fasilitas & Kelistrikan: furnishing, electricity, dll.

Contoh beberapa fitur tambahan hasil rekayasa:

  - building_density = luas bangunan / luas tanah

  - room_ratio = (jumlah kamar + kamar mandi) / jumlah lantai

  - bed_bath_ratio = kamar tidur / kamar mandi

# 🗂️ DATASET

Dataset: jabodetabek_house_price.csv

Dataset terdiri dari 3553 entri dengan 27 kolom dengan atribut:

✳️ Fitur Numerik:
lat, long, bedrooms, bathrooms, land_size_m2, building_size_m2, carports, maid_bedrooms, maid_bathrooms,
floors, building_age, year_built, garages, building_density, room_ratio, bed_bath_ratio.

🔤 Fitur Kategorikal:
district, city, property_type, certificate, furnishing, propety_condition, building_orientation

➕ Fitur Turunan:
building_density, room_ratio, bed_bath_ratio

📌 Gambar: Distribusi Harga Rumah
![Screenshot 2025-07-08 221956](https://github.com/user-attachments/assets/342ccdb8-054e-40c0-bb87-18bfdbe54750)

📌 Gambar: Heatmap Korelasi Fitur Numerik
![image](https://github.com/user-attachments/assets/b071e701-c67d-42cf-b8c1-e5474e0e7080)

# 🔧 PROSES PENYELESAIAN

📌 Gambar: Diagram Alur Model

Langkah-langkah utama:

  1. Import data dan eksplorasi awal.

  2. Bersihkan data dari outlier & missing value.

  3. Feature engineering dan preprocessing (standarisasi dan one-hot encoding).

  4. Split data menjadi data latih dan uji.

  5. Training model Random Forest dan XGBoost.

  6. Evaluasi model dan visualisasi hasil.

# ✨ CUMLIPAN KODE UTAMA

📥 Preprocessing dan Pipeline

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), fitur_numerik),
    ('cat', OneHotEncoder(handle_unknown='ignore'), fitur_kategorikal)
])

🌳 Random Forest Pipeline

rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor())
])

🚀 XGBoost Pipeline

xgb_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(objective='reg:squarederror'))
])

# 📊 EVALUASI PERFORMA
📌 Gambar: Visualisasi Prediksi Random Forest & XGBoost
![Screenshot 2025-07-08 222540](https://github.com/user-attachments/assets/5026bc5d-e7d6-48d4-aa5e-6d1fd266044e)

| Model         | MAE (Rp)    | RMSE (Rp)   | R² Score   |
| ------------- | ----------- | ----------- | --------   |
| Random Forest | 654319962   | 1860495807  | 0.8432     |
| XGBoost       | 637125215   | 1716557060  | 0.8665     |

# ✅ KESIMPULAN
  - Kedua model mampu memprediksi harga rumah dengan cukup baik.

  - XGBoost sedikit lebih unggul dari Random Forest dalam hal akurasi (R² lebih tinggi).

  - Fitur seperti lokasi (city, district), luas tanah, dan jumlah kamar terbukti sangat berpengaruh terhadap harga rumah.
# 🛠️ SARAN PENGEMBANGAN
  - Coba integrasikan data eksternal seperti harga tanah per wilayah atau fasilitas umum terdekat.

  - Gunakan SHAP values untuk interpretasi model lebih dalam.

  - Uji model dengan cross-validation k-fold untuk kestabilan performa.

  - Buat API prediksi sederhana agar bisa digunakan dalam website properti.

---

### 📌 1. Load Data & Eksplorasi Awal
Langkah awal yaitu memuat dataset harga rumah dan melihat struktur umum data: jumlah baris-kolom, tipe data, dan missing value.

```python
import pandas as pd
data = pd.read_csv('jabodetabek_house_price.csv')
print("Ukuran data:", data.shape)
print(data.dtypes)
print(data.isnull().sum())
```

### 📊 2. Visualisasi Distribusi Harga Rumah
Untuk memahami sebaran harga rumah, dilakukan plot histogram baik pada skala asli maupun log.

```python
import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(data['price_in_rp'], bins=50, kde=True)
plt.title('Distribusi Harga Rumah')
plt.show()
```

### 🧠 3. Feature Engineering
Menambahkan fitur baru seperti rasio bangunan dan kamar untuk membantu meningkatkan akurasi prediksi.

```python
data['building_density'] = data['building_size_m2'] / (data['land_size_m2'] + 1)
data['room_ratio'] = (data['bedrooms'] + data['bathrooms']) / (data['floors'] + 1)
data['bed_bath_ratio'] = data['bedrooms'] / (data['bathrooms'] + 1)
```

### 🧼 4. Preprocessing
Standarisasi data numerik dan encoding fitur kategorikal dengan OneHotEncoder.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer([
  ('num', StandardScaler(), fitur_numerik),
  ('cat', OneHotEncoder(handle_unknown='ignore'), fitur_kategorikal)
])
```

### 🌳 5. Random Forest Pipeline
Pipeline untuk Random Forest Regressor yang sudah dilengkapi preprocessing.

```python
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

rf_pipeline = Pipeline([
  ('preprocessor', preprocessor),
  ('model', RandomForestRegressor(random_state=42))
])
```

### ⚡ 6. XGBoost Pipeline
Pipeline untuk XGBoost Regressor, model yang kuat untuk regresi tabular.

```python
from xgboost import XGBRegressor

xgb_pipeline = Pipeline([
  ('preprocessor', preprocessor),
  ('model', XGBRegressor(objective='reg:squarederror', random_state=42))
])
```

### 🔍 7. Grid Search & Training
Pencarian parameter terbaik menggunakan GridSearchCV dan validasi silang (CV).

```python
from sklearn.model_selection import GridSearchCV

param_grid_rf = {
  'model__n_estimators': [200],
  'model__max_depth': [30, None]
}

grid_rf = GridSearchCV(rf_pipeline, param_grid_rf, cv=3, scoring='r2')
grid_rf.fit(X_train, y_train_log)
```

### 📈 8. Evaluasi Model
Prediksi terhadap data uji dan evaluasi performa dengan MAE, RMSE, dan R².

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_pred_rf = np.expm1(grid_rf.predict(X_test))
y_true = np.expm1(y_test)

print("MAE:", mean_absolute_error(y_true, y_pred_rf))
print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred_rf)))
print("R2 Score:", r2_score(y_true, y_pred_rf))
```

### 📊 9. Visualisasi Perbandingan Model
Membandingkan prediksi dari dua model terhadap nilai harga asli.

```python
plt.scatter(y_true, y_pred_rf, label='Random Forest', alpha=0.5)
plt.scatter(y_true, y_pred_xgb, label='XGBoost', alpha=0.5, color='orange')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--')
plt.xlabel('Harga Asli')
plt.ylabel('Harga Prediksi')
plt.title('Perbandingan Prediksi Model')
plt.legend()
plt.grid()
plt.show()
```
