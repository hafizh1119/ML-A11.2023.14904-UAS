🏡 Analisis Komparatif Algoritma Machine Learning untuk Prediksi Harga Rumah Berdasarkan Dataset Jabodetabek

📌 Permasalahan
Bagaimana membangun model prediksi harga rumah yang akurat, efisien, dan dapat diandalkan, serta menentukan model mana yang paling baik melalui evaluasi komparatif berbagai algoritma.

🎯 Tujuan
Mengevaluasi performa algoritma regresi dalam memprediksi harga rumah menggunakan dataset Jabodetabek.
Mengidentifikasi model dengan akurasi terbaik berdasarkan metrik evaluasi seperti RMSE dan R².
Menilai kemampuan masing-masing model dalam merepresentasikan data aktual secara visual dan numerik.
Melakukan analisis residual pada model terbaik untuk memahami pola kesalahan dan potensi penyempurnaan.

🗂️ Struktur Proyek
1. Persiapan Dataset
    from google.colab import files
    uploaded = files.upload()

    import zipfile
    with zipfile.ZipFile("Dataset projek.zip", 'r') as zip_ref:
        zip_ref.extractall("dataset_rumah")

2. Import Library
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.svm import SVR
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_squared_error, r2_score

3. Load Dataset
    df = pd.read_csv('dataset_rumah/jabodetabek_house_price.csv')

4. Pra-pemrosesan Data
    Target (y): price_in_rp → dikonversi ke logaritma.
    Fitur (X): seluruh kolom selain target.
    Pisahkan fitur numerik dan kategorikal.
    Pipeline preprocessing:
        - Numerik → imputasi median.
        - Kategorikal → imputasi modus + OneHotEncoding.

⚙️ Training & Evaluasi Model
    Model yang Digunakan:
    - Random Forest
    - Gradient Boosting
    - Linear Regression
    - Support Vector Regressor (SVR)
    Metode Evaluasi:
    - Mean Squared Error (MSE)
    - R-squared (R²)

📈 Hasil Evaluasi:
    | Model               | MSE    | R²     |
    | ------------------- | ------ | ------ |
    | Random Forest       | 0.1087 | 0.9127 |
    | Gradient Boosting   | 0.1153 | 0.9074 |
    | Linear Regression   | 0.1446 | 0.8762 |
    | Support Vector Reg. | 0.2578 | 0.7540 |
    ✅ Random Forest memberikan hasil paling optimal, mampu menjelaskan 91% variasi data.

🔍 Visualisasi & Analisis
    🔹 Feature Importance (Random Forest)
    Fitur paling berpengaruh:
    1. building_size_m2
    2. land_size_m2
    3. long
    4. lat
    5. bathroom
    6. Luas bangunan (building_size_m2) merupakan faktor paling dominan dalam menentukan harga.

🔹 Prediksi vs Nilai Asli
    - Scatter plot untuk semua model.
    - Garis merah sebagai referensi nilai ideal.
    Random Forest dan Gradient Boosting memiliki prediksi yang mendekati ideal, sedangkan SVR menyebar jauh (underfitting).

🔹 Residual Analysis
    - Visualisasi error (selisih nilai prediksi dan asli) terhadap harga asli.
    Error tersebar merata di sekitar nol menunjukkan model stabil dan tidak bias terhadap harga rendah atau tinggi.

🔹 Distribusi Error
    - Histogram distribusi kesalahan prediksi.
    Random Forest menunjukkan distribusi simetris dan sempit, indikasi model prediksi yang baik.

✅ Kesimpulan
    - Random Forest adalah model terbaik untuk prediksi harga rumah pada dataset ini.
    - Model ini memiliki akurasi tertinggi dan kesalahan terkecil baik secara numerik maupun visual.
    - Luas bangunan adalah fitur paling berpengaruh terhadap harga rumah.

💻 Cara Menjalankan Proyek
    1. Upload file zip dataset (Dataset projek.zip).
    2. Jalankan seluruh cell notebook secara berurutan.
    3. Lihat hasil evaluasi, visualisasi, dan analisis residual.

📁 Dataset
    Dataset digunakan berisi data harga rumah dari berbagai kota di wilayah Jabodetabek, termasuk fitur seperti:
    1. Luas bangunan
    2. Luas tanah
    3. Lokasi (latitude & longitude)
    4. Fasilitas (jumlah kamar, listrik, garasi, dll.)

📊 Tools & Teknologi
    1. Python (Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib)
    2. Google Colab
    3. Machine Learning Regression Models