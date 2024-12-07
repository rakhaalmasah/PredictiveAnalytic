# Laporan Proyek Machine Learning - [Muhammad Rakha Almasah]

## Domain Proyek

### **Latar Belakang**
Pasar saham adalah indikator vital kesehatan ekonomi suatu negara. Analisis dan prediksi pergerakan indeks saham, seperti NASDAQ dan S&P 500 (SPY), membantu investor dalam pengambilan keputusan yang lebih baik. Namun, dinamika pergerakan saham sangat kompleks, dipengaruhi oleh berbagai faktor, termasuk kinerja perusahaan besar seperti Apple, Microsoft, Amazon, dan Berkshire Hathaway.

Penelitian sebelumnya telah mengeksplorasi berbagai pendekatan untuk memprediksi pasar saham. Misalnya, Shah et al. (2021) menggunakan prediksi berbasis berita dengan text mining dan machine learning untuk meningkatkan akurasi prediksi pasar keuangan. Selain itu, Kim et al. (2020) menerapkan algoritma hybrid machine learning untuk mengidentifikasi arah pergerakan harian pasar saham dengan lebih akurat. Lebih lanjut, Huang et al. (2021) membahas aplikasi machine learning dalam peramalan keuangan, perencanaan, dan analisis, sementara Gupta et al. (2023) menganalisis pengaruh kecerdasan buatan dan machine learning terhadap pasar keuangan, termasuk dalam trading, manajemen risiko, dan operasi keuangan.

Prediksi indeks saham dengan machine learning menawarkan potensi besar dalam memberikan wawasan bagi investor dan analis keuangan untuk memahami pola pasar yang kompleks.

### **Referensi**
- [News-Based Intelligent Prediction of Financial Markets Using Text Mining and Machine Learning: A Systematic Literature Review](https://scholar.google.com/scholar?hl=id&as_sdt=0%2C5&q=News-Based+Intelligent+Prediction+of+Financial+Markets+Using+Text+Mining+and+Machine+Learning%3A+A+Systematic+Literature+Review&btnG=)

- [Predicting the Daily Return Direction of the Stock Market Using Hybrid Machine Learning Algorithms](https://scholar.google.com/scholar?hl=id&as_sdt=0%2C5&q=Predicting+the+Daily+Return+Direction+of+the+Stock+Market+Using+Hybrid+Machine+Learning+Algorithms+&btnG=)
  
- [Machine learning for financial forecasting, planning and analysis](https://link.springer.com/article/10.1007/s42521-021-00046-2)

- [Unveiling the Influence of Artificial Intelligence and Machine Learning on Financial Markets: A Comprehensive Analysis of AI Applications in Trading, Risk Management, and Financial Operations](https://www.mdpi.com/1911-8074/16/10/434)

  
## Business Understanding

### **Problem Statements**
1. Bagaimana membangun model machine learning untuk memprediksi nilai indeks saham NASDAQ dan SPY berdasarkan data historis perusahaan besar?
2. Algoritma machine learning apa yang memberikan prediksi paling akurat?
3. Bagaimana cara meningkatkan akurasi prediksi melalui teknik hyperparameter tuning?

### **Goals**
1. Mengembangkan model machine learning untuk memprediksi NASDAQ dan SPY berdasarkan data saham Apple, Microsoft, Amazon, dan Berkshire Hathaway.
2. Membandingkan performa algoritma seperti Random Forest, XGBoost, dan LightGBM dalam memprediksi nilai indeks saham.
3. Melakukan hyperparameter tuning untuk meningkatkan akurasi model.

### **Solution Statements**
1. Menggunakan berbagai algoritma seperti Random Forest, XGBoost, LightGBM, Gradient Boosting, dan CatBoost untuk memprediksi NASDAQ dan SPY.
2. Melakukan evaluasi model menggunakan metrik Mean Squared Error (MSE).
3. Meningkatkan performa model melalui GridSearchCV untuk menemukan hyperparameter terbaik.

---

## Data Preparation and Understanding

### 1. Gathering Data

#### Proses Pengumpulan Data
Data diambil dari [Yahoo Finance](https://finance.yahoo.com/) menggunakan library `yfinance`. Berikut langkah-langkah pengumpulan data:
1. **Instalasi dan Impor Library**:
   - Library `yfinance` digunakan untuk mengunduh data historis.
   - Library lain seperti `numpy`, `pandas`, `matplotlib`, dan `seaborn` digunakan untuk analisis dan visualisasi data.
2. **Rentang Waktu**:
   - Data dikumpulkan dari 1 Januari 2001 hingga 1 Desember 2024.
3. **Target (Variabel Y)**:
   - `Close_nasdaq`: Harga penutupan NASDAQ Composite.
   - `Close_spy`: Harga penutupan S&P 500 ETF.
4. **Fitur (Variabel X)**:
   - `Close_aapl`: Harga penutupan saham Apple.
   - `Close_msft`: Harga penutupan saham Microsoft.
   - `Close_amzn`: Harga penutupan saham Amazon.
   - `Close_brkb`: Harga penutupan saham Berkshire Hathaway.
5. **Verifikasi Data**:
   - Dataset yang diunduh diverifikasi dengan menampilkan beberapa baris pertama untuk memastikan data sesuai.

---

### 2. Cleaning and Processing Data

#### Proses Pembersihan Data
1. **Penilaian Dataset**:
   - Setiap dataset diperiksa untuk memastikan tidak ada nilai hilang (*missing values*) atau duplikasi.
   - Dataset memiliki kolom utama: `Date` dan `Close`, di mana `Date` perlu dikonversi ke tipe datetime.
2. **Deteksi Outliers**:
   - Menggunakan metode *Interquartile Range (IQR)* untuk mendeteksi nilai ekstrem di luar batas wajar.
   - Nilai di luar batas bawah dan atas (*lower bound* dan *upper bound*) dihapus untuk meningkatkan kualitas data.
3. **Penggabungan Dataset**:
   - Semua dataset digabung berdasarkan kolom `Date` menggunakan *inner join* untuk memastikan keselarasan waktu antar fitur dan target.

#### Hasil Pembersihan Data
Dataset hasil penggabungan berisi 4954 baris dan 7 kolom, yang siap digunakan untuk analisis prediktif.

---

### 3. Data Understanding

#### Informasi Dataset
- **Jumlah Baris**: 4954
- **Jumlah Kolom**: 7
- **Kolom Dataset**:
  1. **Date**: Tanggal pengambilan data.
  2. **Close_nasdaq**: Harga penutupan NASDAQ Composite.
  3. **Close_spy**: Harga penutupan S&P 500 ETF.
  4. **Close_aapl**: Harga penutupan saham Apple.
  5. **Close_msft**: Harga penutupan saham Microsoft.
  6. **Close_amzn**: Harga penutupan saham Amazon.
  7. **Close_brkb**: Harga penutupan saham Berkshire Hathaway.

**Sumber Data**: [Yahoo Finance](https://finance.yahoo.com/)  
**Tautan Dataset Gabungan**: [Dataset Ready-to-Use](https://github.com/rakhaalmasah/PredictiveAnalytic/blob/dbf7d3817526368333d68d06785f15868c5e91b2/Dataset/merged_dataset.csv)

---

#### Univariate Analysis
1. **Target (Close_nasdaq dan Close_spy)**:
   - `Close_nasdaq`: Volatilitas tinggi dengan rata-rata 3664.91, rentang 1114.11 hingga 11880.63, dan distribusi *right-skewed*.
   - `Close_spy`: Stabil dengan rata-rata 163.64 dan rentang 68.11 hingga 357.46.

2. **Fitur Numerik (Close_aapl, Close_msft, Close_amzn, Close_brkb)**:
   - `Close_aapl` dan `Close_amzn`: Distribusi *right-skewed* dengan beberapa nilai ekstrem sebagai outliers.
   - `Close_msft`: Distribusi lebih merata dibandingkan Apple dan Amazon.
   - `Close_brkb`: Distribusi paling stabil.

---

#### Multivariate Analysis
1. **Korelasi Antar Variabel**:
   - Semua fitur numerik memiliki hubungan positif sangat kuat dengan target:
     - Contoh: `Close_aapl` dan `Close_nasdaq` (0.97).
   - NASDAQ dan SPY memiliki korelasi sangat kuat (0.99), mencerminkan pola pergerakan pasar yang selaras.

2. **Scatterplot dan Pairplot**:
   - Scatterplot menunjukkan pola linier positif antara fitur dan target.
   - Pairplot mempertegas hubungan antar variabel dan menunjukkan keberadaan outliers pada beberapa saham teknologi.

---

#### Insight dari Data Understanding
1. **Hubungan antar Variabel**:
   - Saham teknologi memiliki kontribusi besar terhadap pergerakan NASDAQ.
   - Saham Berkshire Hathaway lebih stabil, mencerminkan perbedaan sektor industri.
2. **Distribusi Data**:
   - NASDAQ dan saham teknologi menunjukkan volatilitas tinggi.
   - SPY dan Berkshire Hathaway lebih stabil, mencerminkan pasar yang lebih luas.
3. **Kualitas Data**:
   - Data telah dibersihkan dari outliers dan tidak memiliki nilai hilang atau duplikasi.

---

### 4. Kesimpulan
Proses pengumpulan, pembersihan, dan eksplorasi data menghasilkan dataset yang berkualitas dan siap digunakan untuk analisis prediktif. Dataset ini mencerminkan pola volatilitas pasar saham teknologi dan stabilitas sektor lain, memberikan fondasi yang kuat untuk membangun model prediktif.


---

## Data Preparation

### **Proses Data Preparation**
1. **Pemisahan Target dan Fitur:** Target: `Close_nasdaq` dan `Close_spy`. Fitur: `Close_aapl`, `Close_msft`, `Close_amzn`, dan `Close_brkb`.
2. **Split Dataset:** Dataset dibagi menjadi 90% data latih dan 10% data uji menggunakan `train_test_split`.
3. **Standarisasi Data:** Fitur numerik distandarisasi menggunakan `StandardScaler` untuk meningkatkan performa algoritma.

### **Alasan Data Preparation**
- **Pemisahan target dan fitur** memastikan bahwa model hanya menggunakan data input untuk prediksi.
- **Standarisasi data** diperlukan untuk algoritma yang sensitif terhadap skala fitur, seperti SVR.

---

## Modeling

### **Algoritma yang Digunakan**
1. **Random Forest:** Algoritma ensemble berbasis pohon keputusan yang robust terhadap overfitting.
2. **AdaBoost:** Memperbaiki kesalahan model sebelumnya secara iteratif.
3. **SVR:** Algoritma berbasis Support Vector Machine untuk regresi.
4. **Gradient Boosting:** Teknik boosting untuk menangkap pola non-linear.
5. **XGBoost:** Implementasi Gradient Boosting yang dioptimalkan.
6. **LightGBM:** Model boosting yang efisien untuk dataset besar.
7. **CatBoost:** Model boosting yang menangani data kategori dengan baik.
8. **ElasticNet:** Kombinasi regularisasi L1 dan L2 untuk regresi linier.

### **Hyperparameter Tuning**
Setiap model dioptimalkan menggunakan `GridSearchCV`. Parameter seperti `n_estimators`, `learning_rate`, dan `max_depth` diuji untuk menemukan kombinasi terbaik.

### **Hasil Evaluasi Model**
| Model             | Train MSE | Test MSE | Best Params                                |
|--------------------|-----------|----------|-------------------------------------------|
| Random Forest      | 0.337     | 1.952    | {'max_depth': 20, 'n_estimators': 150}    |
| XGBoost            | 0.363     | 2.337    | {'learning_rate': 0.2, 'n_estimators': 100}|
| LightGBM           | 0.832     | 2.113    | {'learning_rate': 0.2, 'n_estimators': 100}|
| Gradient Boosting  | 2.455     | 4.214    | {'learning_rate': 0.2, 'n_estimators': 100}|
| CatBoost           | 2.321     | 3.319    | {'iterations': 150, 'learning_rate': 0.2} |
| AdaBoost           | 32.244    | 35.258   | {'learning_rate': 0.2, 'n_estimators': 100}|
| SVR                | 30.100    | 31.615   | {'C': 10, 'kernel': 'linear'}             |
| ElasticNet         | 29.995    | 31.559   | {'alpha': 0.1, 'l1_ratio': 0.5}           |

- **Model Terbaik:** Random Forest dengan Test MSE terendah (1.952).

---

## Evaluation

### **Metrik Evaluasi**
Metrik yang digunakan adalah **Mean Squared Error (MSE)**:
\[
\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
\]

- **MSE pada Random Forest:**
  - **Train MSE:** 0.337
  - **Test MSE:** 1.952

### **Visualisasi Prediksi**
1. **Prediksi NASDAQ:** Prediksi sangat dekat dengan nilai aktual, menunjukkan performa model yang sangat baik.
2. **Prediksi SPY:** Model juga memprediksi nilai SPY dengan akurasi tinggi, sebagaimana ditunjukkan oleh scatter plot.

### **Kesimpulan**
- Model Random Forest berhasil memprediksi NASDAQ dan SPY dengan akurasi tinggi, menjadikannya model terbaik untuk proyek ini.
- Hyperparameter tuning berkontribusi dalam meningkatkan akurasi model, terutama dibandingkan dengan model baseline.

---

## Catatan Tambahan
- **Peluang Perbaikan:**
  - Penambahan fitur lain yang relevan (contoh: data ekonomi makro).
  - Eksplorasi algoritma lain, seperti deep learning, untuk dataset yang lebih besar.
- **Implementasi:** Model dapat digunakan sebagai alat prediksi untuk membantu investor memahami pola pasar saham.

