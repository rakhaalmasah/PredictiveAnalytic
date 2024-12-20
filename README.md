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

## Data Understanding

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
##### Variabel Target: `Close_nasdaq` dan `Close_spy`
1. **Close_nasdaq**:
   - **Statistik**:
     - Rata-rata harga penutupan: **3664.91**
     - Rentang harga: **1114.11 hingga 11880.63**
     - Distribusi miring ke kanan (*right-skewed*), menunjukkan adanya beberapa harga sangat tinggi dibandingkan dengan mayoritas.
   - **Insight Utama**:
     - Sebagian besar harga penutupan berada pada rentang **2000 hingga 4000**.
     - Harga di atas **5000** mencerminkan puncak kinerja pasar saham teknologi pada periode tertentu.

2. **Close_spy**:
   - **Statistik**:
     - Rata-rata harga penutupan: **163.64**
     - Rentang harga: **68.11 hingga 357.46**
     - Distribusi lebih stabil dibandingkan NASDAQ, dengan pola yang mendekati simetris.
   - **Insight Utama**:
     - Sebagian besar harga terkonsentrasi pada dua rentang utama: **100 hingga 150** dan **200 hingga 250**, mencerminkan perubahan signifikan dalam kinerja pasar secara keseluruhan.
     - Stabilitas SPY mencerminkan perannya sebagai indeks pasar yang mencakup berbagai sektor.

---

##### Fitur Numerik: `Close_aapl`, `Close_msft`, `Close_amzn`, `Close_brkb`
1. **Close_aapl (Apple)**:
   - **Statistik**:
     - Rata-rata harga: **18.02**
     - Rentang harga: **0.23 hingga 114.61**
     - Distribusi miring ke kanan (*right-skewed*), dengan sebagian besar harga di bawah **20**.
   - **Insight Utama**:
     - Lonjakan harga yang tinggi mencerminkan peristiwa penting seperti *stock split* atau inovasi teknologi yang signifikan.

2. **Close_msft (Microsoft)**:
   - **Statistik**:
     - Rata-rata harga: **47.44**
     - Rentang harga: **15.15 hingga 216.54**
     - Distribusi lebih merata dibandingkan Apple, dengan sebagian besar harga berada pada rentang **25 hingga 50**.
   - **Insight Utama**:
     - Stabilitas Microsoft mencerminkan pertumbuhan konsisten dan kehadiran pasar yang kuat.

3. **Close_amzn (Amazon)**:
   - **Statistik**:
     - Rata-rata harga: **22.52**
     - Rentang harga: **0.30 hingga 161.25**
     - Distribusi sangat miring ke kanan (*right-skewed*), dengan sebagian besar harga di bawah **20**.
   - **Insight Utama**:
     - Lonjakan harga mencerminkan perkembangan signifikan dalam inovasi e-commerce dan bisnis.

4. **Close_brkb (Berkshire Hathaway)**:
   - **Statistik**:
     - Rata-rata harga: **103.14**
     - Rentang harga: **40 hingga 230.20**
     - Distribusi paling stabil di antara saham lainnya, terkonsentrasi di sekitar **50 hingga 150**.
   - **Insight Utama**:
     - Stabilitas mencerminkan pendekatan investasi jangka panjang dan strategi pasar yang konservatif.
  
---

#### Multivariate Analysis
##### Korelasi Antar Variabel
1. **Korelasi dengan Variabel Target**:
   - Semua fitur numerik menunjukkan **korelasi positif yang sangat kuat** dengan `Close_nasdaq` dan `Close_spy`:
     - Contoh: `Close_aapl` memiliki korelasi **0.97** dengan `Close_nasdaq` dan **0.93** dengan `Close_spy`.
   - **Insight Utama**:
     - Saham teknologi (Apple, Microsoft, Amazon) memiliki pengaruh besar terhadap indeks NASDAQ.
     - Berkshire Hathaway menunjukkan hubungan yang lebih kuat dengan S&P 500 dibandingkan NASDAQ.

2. **Korelasi Antar Fitur Numerik**:
   - Saham teknologi (Apple, Microsoft, Amazon) memiliki **korelasi yang sangat tinggi** satu sama lain (di atas **0.9**).
   - Berkshire Hathaway memiliki korelasi yang lebih rendah dengan saham teknologi tetapi tetap positif.
   - **Insight Utama**:
     - Korelasi kuat di antara saham teknologi mencerminkan pergerakan yang terkoordinasi akibat tren global atau peristiwa industri.

3. **Korelasi Antara `Close_nasdaq` dan `Close_spy`**:
   - Korelasi **0.99** menunjukkan bahwa kedua indeks ini bergerak hampir identik.
   - **Insight Utama**:
     - Meskipun NASDAQ berfokus pada teknologi, hubungan kuat dengan S&P 500 mencerminkan pengaruh teknologi terhadap pasar secara keseluruhan.

---

##### Analisis Scatterplot dan Pairplot
1. **Insight dari Scatterplot**:
   - Hubungan linier positif yang kuat terlihat antara fitur numerik dan variabel target.
   - Contoh: `Close_aapl` dan `Close_nasdaq` menunjukkan hubungan linier yang jelas, mengindikasikan pengaruh signifikan Apple terhadap NASDAQ.
   - **Insight Utama**:
     - Pola linier menunjukkan bahwa model regresi dapat menangkap hubungan ini dengan baik.

2. **Insight dari Pairplot**:
   - Pairplot mengonfirmasi pola yang terlihat pada scatterplot, dengan plot KDE pada diagonal menunjukkan:
     - Distribusi miring ke kanan (*right-skewed*) untuk saham teknologi (Apple, Amazon, Microsoft).
     - Distribusi lebih simetris untuk Berkshire Hathaway.
   - **Insight Utama**:
     - Pairplot membantu mengidentifikasi outlier pada saham teknologi, khususnya Apple dan Amazon.

3. **Outlier**:
   - Scatterplot dan pairplot mengungkap keberadaan outlier pada saham teknologi (contoh: harga tinggi pada Apple dan Amazon).
   - **Insight Utama**:
     - Outlier ini kemungkinan besar terkait dengan peristiwa pasar yang signifikan atau inovasi teknologi.


---

###### Insight dari Data Understanding
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
Berikut adalah algoritma machine learning yang digunakan dalam proyek ini:

1. **Random Forest:** Algoritma ensemble berbasis pohon keputusan yang robust terhadap overfitting.
2. **AdaBoost:** Algoritma boosting untuk meningkatkan akurasi model iteratif.
3. **SVR:** Algoritma berbasis Support Vector Machine untuk regresi.
4. **Gradient Boosting:** Teknik boosting untuk menangkap pola non-linear.
5. **XGBoost:** Implementasi Gradient Boosting yang lebih cepat dan efisien.
6. **LightGBM:** Model boosting yang dioptimalkan untuk dataset besar.
7. **CatBoost:** Model boosting yang menangani data kategori dengan baik.
8. **ElasticNet:** Kombinasi regularisasi L1 dan L2 untuk regresi linier.

---

### **Hyperparameter Tuning**
Setiap model dioptimalkan menggunakan `GridSearchCV`. Parameter yang diuji meliputi:
- `n_estimators` (jumlah pohon/iterasi).
- `learning_rate` (langkah iterasi).
- Parameter tambahan spesifik untuk setiap algoritma (seperti `max_depth`, `alpha`, dan `l1_ratio`).

### **Penjelasan Parameter Model**

#### **1. Random Forest**
- **Parameter Dicoba:**
  - `n_estimators`: [50, 100, 150]
  - `max_depth`: [10, 20, 30]
- **Parameter Terpilih:** `n_estimators=150`, `max_depth=20`
- **Kelebihan:** Mampu menangkap pola data yang kompleks.
- **Kekurangan:** Komputasi berat untuk dataset besar.
- **Hasil:** `Test MSE=1.952` (model terbaik).

---

#### **2. AdaBoost**
- **Parameter Dicoba:**
  - `n_estimators`: [50, 100, 150]
  - `learning_rate`: [0.01, 0.1, 0.2]
- **Parameter Terpilih:** `n_estimators=100`, `learning_rate=0.2`
- **Kelebihan:** Fokus pada data sulit.
- **Kekurangan:** Sensitif terhadap outliers.
- **Hasil:** `Test MSE=35.258` (underfitting).

---

#### **3. SVR**
- **Parameter Dicoba:**
  - `C`: [0.1, 1, 10]
  - `kernel`: ['linear', 'rbf']
- **Parameter Terpilih:** `C=10`, `kernel=linear`
- **Kelebihan:** Bagus untuk dataset kecil.
- **Kekurangan:** Tidak cocok untuk dataset besar.
- **Hasil:** `Test MSE=31.615` (underfitting).

---

#### **4. Gradient Boosting**
- **Parameter Dicoba:**
  - `n_estimators`: [50, 100, 150]
  - `learning_rate`: [0.01, 0.1, 0.2]
- **Parameter Terpilih:** `n_estimators=100`, `learning_rate=0.2`
- **Kelebihan:** Menangkap pola non-linear.
- **Kekurangan:** Komputasi lambat.
- **Hasil:** `Test MSE=4.214`.

---

#### **5. XGBoost**
- **Parameter Dicoba:**
  - `n_estimators`: [50, 100, 150]
  - `learning_rate`: [0.01, 0.1, 0.2]
- **Parameter Terpilih:** `n_estimators=100`, `learning_rate=0.2`
- **Kelebihan:** Cepat dan efisien.
- **Kekurangan:** Relatif kompleks.
- **Hasil:** `Test MSE=2.337`.

---

#### **6. LightGBM**
- **Parameter Dicoba:**
  - `n_estimators`: [50, 100, 150]
  - `learning_rate`: [0.01, 0.1, 0.2]
- **Parameter Terpilih:** `n_estimators=100`, `learning_rate=0.2`
- **Kelebihan:** Efisien untuk dataset besar.
- **Kekurangan:** Sensitif terhadap outliers.
- **Hasil:** `Test MSE=2.113`.

---

#### **7. CatBoost**
- **Parameter Dicoba:**
  - `iterations`: [50, 100, 150]
  - `learning_rate`: [0.01, 0.1, 0.2]
- **Parameter Terpilih:** `iterations=150`, `learning_rate=0.2`
- **Kelebihan:** Menangani data kategori tanpa encoding.
- **Kekurangan:** Komputasi kompleks.
- **Hasil:** `Test MSE=3.319`.

---

#### **8. ElasticNet**
- **Parameter Dicoba:**
  - `alpha`: [0.1, 0.5, 1.0]
  - `l1_ratio`: [0.2, 0.5, 0.8]
- **Parameter Terpilih:** `alpha=0.1`, `l1_ratio=0.5`
- **Kelebihan:** Stabil untuk dataset dengan multikolinearitas.
- **Kekurangan:** Tidak cocok untuk hubungan non-linear.
- **Hasil:** `Test MSE=31.559`.

---

#### **Kesimpulan**
1. **Model Terbaik: Random Forest**
   - `Test MSE=1.952`, menangkap pola dengan baik tanpa overfitting.
2. **Model Alternatif: LightGBM dan XGBoost**
   - LightGBM (`Test MSE=2.113`) dan XGBoost (`Test MSE=2.337`) juga memiliki performa sangat baik.
3. **Model Kurang Cocok:**
   - ElasticNet dan SVR gagal menangkap hubungan non-linear, sementara AdaBoost underfitting pada data kompleks.

---

## Evaluation

### **Metrik Evaluasi**

Metrik yang digunakan untuk mengevaluasi model adalah **Mean Squared Error (MSE)**. Rumus untuk menghitung MSE diberikan sebagai berikut:

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

Dimana:
- **yᵢ**: Nilai aktual (true values).
- **ŷᵢ**: Nilai prediksi (predicted values).
- **n**: Jumlah data.

Nilai **MSE** menggambarkan rata-rata dari kuadrat kesalahan antara nilai prediksi dan nilai aktual. Semakin kecil nilai **MSE**, semakin baik model dalam memprediksi data.

---
### **Hasil Evaluasi Model**
| **Model**         | **Train MSE** | **Test MSE** | **Best Params**                                |
|--------------------|---------------|--------------|-----------------------------------------------|
| Random Forest      | 0.337         | 1.952        | {'n_estimators': 150, 'max_depth': 20}        |
| XGBoost            | 0.363         | 2.337        | {'n_estimators': 100, 'learning_rate': 0.2}   |
| LightGBM           | 0.832         | 2.113        | {'n_estimators': 100, 'learning_rate': 0.2}   |
| Gradient Boosting  | 2.455         | 4.214        | {'n_estimators': 100, 'learning_rate': 0.2}   |
| CatBoost           | 2.321         | 3.319        | {'iterations': 150, 'learning_rate': 0.2}     |
| AdaBoost           | 32.244        | 35.258       | {'n_estimators': 100, 'learning_rate': 0.2}   |
| SVR                | 30.100        | 31.615       | {'C': 10, 'kernel': 'linear'}                 |
| ElasticNet         | 29.995        | 31.559       | {'alpha': 0.1, 'l1_ratio': 0.5}               |
---

### **Hasil Evaluasi MSE pada Random Forest**
1. **Train MSE:** 0.337
2. **Test MSE:** 1.952

Model **Random Forest** menunjukkan nilai **MSE** terendah pada data uji, yang mengindikasikan performa prediksi terbaik dibandingkan model lain.

---

### **Visualisasi Prediksi**

#### **Prediksi NASDAQ (Close_nasdaq):**
- Prediksi model Random Forest sangat dekat dengan nilai aktual, menunjukkan bahwa model mampu menangkap pola data dengan baik.
- Model memprediksi nilai NASDAQ tanpa mengalami overfitting pada data latih.

##### Grafik Prediksi NASDAQ:
![Prediksi NASDAQ](https://raw.githubusercontent.com/rakhaalmasah/PredictiveAnalytic/b9b35e8e83fe1917f19af83538543f1a611897e3/nasdaq.png)

---

#### **Prediksi SPY (Close_spy):**
- Model juga menunjukkan performa prediksi yang sangat baik untuk data SPY, sebagaimana terlihat dari scatter plot prediksi vs nilai aktual.

##### Grafik Prediksi SPY:
![Prediksi SPY](https://raw.githubusercontent.com/rakhaalmasah/PredictiveAnalytic/b9b35e8e83fe1917f19af83538543f1a611897e3/snp.png)

---

### Evaluasi Dampak Model terhadap Business Understanding

#### **1. Apakah sudah menjawab problem statement?**
**Ya**, model yang dikembangkan telah berhasil menjawab problem statement yang dirumuskan:
1. **Problem Statement 1**: 
   - *"Bagaimana membangun model machine learning untuk memprediksi nilai indeks saham NASDAQ dan SPY berdasarkan data historis perusahaan besar?"*
   - **Jawaban:** Model **Random Forest** berhasil memprediksi nilai indeks **NASDAQ** dan **SPY** dengan akurasi tinggi yang ditandai dengan Test MSE rendah yaitu 1.952. Data historis saham perusahaan besar seperti Apple, Microsoft, Amazon, dan Berkshire Hathaway memberikan kontribusi besar terhadap prediksi ini.

2. **Problem Statement 2**: 
   - *"Algoritma machine learning apa yang memberikan prediksi paling akurat?"*
   - **Jawaban:** Berdasarkan evaluasi metrik **Mean Squared Error (MSE)**, **Random Forest** adalah algoritma terbaik dengan Test MSE terendah (1.952), diikuti oleh **LightGBM** (2.113) dan **XGBoost** (2.337).

3. **Problem Statement 3**: 
   - *"Bagaimana cara meningkatkan akurasi prediksi melalui teknik hyperparameter tuning?"*
   - **Jawaban:** Hyperparameter tuning menggunakan GridSearchCV menghasilkan parameter optimal untuk model Random Forest (`max_depth: 20, n_estimators: 150`), meningkatkan performa model tanpa overfitting.

---

#### **2. Apakah berhasil mencapai goals yang diharapkan?**
**Ya**, semua goals tercapai:
1. **Goal 1:** Mengembangkan model machine learning untuk memprediksi NASDAQ dan SPY berdasarkan data saham perusahaan besar.
   - Model berhasil dibangun dan menghasilkan prediksi yang sangat akurat untuk **Close_nasdaq** dan **Close_spy** yang ditandai dengan Test MSE terendah (1.952).

2. **Goal 2:** Membandingkan performa algoritma seperti Random Forest, XGBoost, dan LightGBM.
   - Evaluasi menunjukkan bahwa **Random Forest** adalah model terbaik, diikuti oleh **LightGBM** dan **XGBoost**.

3. **Goal 3:** Melakukan hyperparameter tuning untuk meningkatkan akurasi model.
   - Hyperparameter tuning berhasil meningkatkan performa model, menjadikan Random Forest model dengan prediksi terbaik.

---

#### **3. Apakah solusi statement yang direncanakan berdampak?**
**Ya**, solusi yang direncanakan memiliki dampak nyata:
1. **Relevansi dengan Prediksi Pasar Saham:**
   - Solusi berbasis machine learning menyediakan alat prediktif andal yang dapat digunakan investor untuk memahami tren pasar saham, khususnya NASDAQ dan SPY.

2. **Akurasi Tinggi:** 
   - Model **Random Forest** menghasilkan prediksi yang sangat akurat, sehingga meningkatkan kepercayaan investor dalam menggunakan model ini untuk strategi perdagangan saham.

3. **Dampak terhadap Investasi:**
   - Model ini memberikan wawasan tentang tren pasar yang didorong oleh sektor teknologi, sehingga membantu investor mengidentifikasi peluang investasi dengan lebih baik.

4. **Pengambilan Keputusan yang Lebih Informed:**
   - Solusi ini memungkinkan keputusan investasi berbasis data (*data-driven decision-making*), mengurangi ketidakpastian, dan meningkatkan akurasi strategi investasi.

---

### **Kesimpulan**
1. Model **Random Forest** terbukti menjadi model terbaik untuk memprediksi **Close_nasdaq** dan **Close_spy**, dengan **MSE Test** terendah sebesar **1.952**.
2. **Hyperparameter Tuning** melalui GridSearchCV berkontribusi dalam meningkatkan akurasi prediksi model.
3. Visualisasi prediksi memperlihatkan bahwa model mampu menghasilkan prediksi yang sangat mendekati nilai aktual.

---

## Catatan Tambahan
- **Peluang Perbaikan:**
  - Penambahan fitur lain yang relevan (contoh: data ekonomi makro).
  - Eksplorasi algoritma lain, seperti deep learning, untuk dataset yang lebih besar.
- **Implementasi:** Model dapat digunakan sebagai alat prediksi untuk membantu investor memahami pola pasar saham.

