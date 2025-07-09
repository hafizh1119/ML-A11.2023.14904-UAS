# 🏡 PREDIKSI HARGA RUMAH DI JABODETABEK MENGGUNAKAN RANDOM FOREST & XGBOOST

Nama : Hafizh Naufal Nuha Kusuma

NIM  : A11.2023.14904

Klmpk: A11.4406

---

Proyek ini berawal dari tantangan teknis dalam membangun model prediksi harga rumah yang akurat di wilayah Jabodetabek. Harga rumah dipengaruhi oleh banyak fitur numerik dan kategorikal seperti lokasi, ukuran bangunan, jenis sertifikat, kondisi rumah, dan lainnya.

Permasalahan utama dalam membangun model prediksi adalah:

1. Sebaran harga rumah yang sangat skewed (condong kanan), yang membuat model kesulitan menghasilkan prediksi stabil tanpa transformasi logaritmik.

2. Banyaknya fitur kategorikal yang perlu di-encode agar bisa digunakan dalam model berbasis pohon keputusan.

3. Adanya missing value dan outlier dalam data historis properti yang berpotensi mengganggu performa model.

4. Kebutuhan untuk membandingkan dua algoritma yang kuat, yaitu Random Forest dan XGBoost, untuk mengetahui mana yang lebih akurat dan efisien dalam konteks regresi harga rumah.
---
# 🎯 TUJUAN PROYEK
  Proyek ini bertujuan untuk mengembangkan dan membandingkan dua model regresi (Random Forest dan XGBoost) dalam memprediksi harga rumah di wilayah Jabodetabek, dengan fokus pada aspek teknis berikut:

  1. Melakukan eksplorasi data dan rekayasa fitur untuk menghasilkan representasi data yang optimal bagi model prediksi.

  2. Menerapkan preprocessing yang sesuai untuk menangani data numerik dan kategorikal secara bersamaan menggunakan ColumnTransformer.

  3. Melakukan transformasi target log1p untuk mengatasi sebaran harga yang skewed dan meningkatkan stabilitas model.

  4. Menghapus outlier dan melakukan feature engineering tambahan seperti building_density, room_ratio, dan bed_bath_ratio untuk menambah informasi struktural properti.

  5. Melakukan hyperparameter tuning (GridSearchCV) untuk mendapatkan kombinasi parameter terbaik pada masing-masing model.

  6. Membandingkan performa model berdasarkan metrik MAE, RMSE, dan R² Score, serta melakukan visualisasi selisih prediksi vs harga asli.

  7. Menentukan model terbaik secara empiris, berdasarkan hasil evaluasi pada data uji, untuk digunakan sebagai sistem prediksi harga rumah berbasis data historis.
# 🔄 Alur Penyelesaian Proyek

![Data Set (1)](https://github.com/user-attachments/assets/e938047e-ff02-4a20-990f-72863b816e4c)

1. Input Data
   
    Dataset yang digunakan adalah data harga rumah di wilayah Jabodetabek, yang mencakup fitur-fitur numerik seperti bedrooms, land_size_m2, serta fitur kategorikal seperti city, certificate, dan lainnya.
   
2. Eksplorasi Data (EDA)
   
   Tahap ini melibatkan analisis awal untuk memahami struktur data, distribusi harga (price_in_rp), deteksi outlier, serta missing values. Termasuk visualisasi seperti histogram dan heatmap korelasi fitur numerik.
   
3. Feature Engineering & Preprocessing

   - Pembuatan fitur turunan seperti building_density, room_ratio, dan bed_bath_ratio.

   - Penanganan missing value (menggunakan fillna)

   - Normalisasi fitur numerik (StandardScaler) dan encoding fitur kategorikal (OneHotEncoder) menggunakan ColumnTransformer.
    
4. Pemodelan Machine Learning
   
   - Random Forest Regressor menggunakan pipeline dan GridSearchCV untuk tuning.

   - XGBoost Regressor juga diproses dalam pipeline serupa, dengan fokus pada performa dan efisiensi.
     
5. Evaluasi Model

   - MAE (Mean Absolute Error)

   - RMSE (Root Mean Squared Error)

   - R² Score

     Target prediksi dilakukan terhadap variabel log(price_in_rp) yang kemudian dikembalikan ke skala asli dengan expm1.
     
6. Visualisasi & Interpretasi Hasil

   - Scatter plot prediksi vs harga asli

   - Histogram selisih dan persentase kesalahan

   - Feature importance (untuk XGBoost)
   
     Visualisasi ini membantu memahami seberapa akurat model memetakan fitur terhadap harga sebenarnya.

# 📚 Dataset

Dataset: jabodetabek_house_price.csv

Dataset terdiri dari 3553 entri dengan 27 kolom dengan atribut:

✳️ Fitur Numerik:

    lat, long, bedrooms, bathrooms, land_size_m2, building_size_m2, carports, maid_bedrooms, maid_bathrooms, floors, building_age, year_built, garages, building_density, room_ratio, bed_bath_ratio.

🔤 Fitur Kategorikal:

    district, city, property_type, certificate, furnishing, propety_condition, building_orientation

➕ Fitur Turunan:

    building_density, room_ratio, bed_bath_ratio

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

📌 Gambar: Distribusi Harga Rumah
![Screenshot 2025-07-08 221956](https://github.com/user-attachments/assets/342ccdb8-054e-40c0-bb87-18bfdbe54750)

📌 Gambar: Heatmap Korelasi Fitur Numerik
![image](https://github.com/user-attachments/assets/b071e701-c67d-42cf-b8c1-e5474e0e7080)

# ✨ Proses Learning dan Modeling

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
