from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

# =========================================
# ANALISIS HASIL MODEL
# =========================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("===== MODEL ANALYSIS =====")

# ================================
# 1. SCATTER PLOT
# z_spec vs z_pred
# ================================

plt.figure(figsize=(6,6))

plt.scatter(y_test, y_pred, s=5)

plt.xlabel("Spectroscopic Redshift (z_spec)")
plt.ylabel("Predicted Redshift (z_phot)")
plt.title("Photometric Redshift Prediction")

# garis ideal
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="red")

plt.show()

plt.savefig("z_spec_vs_z_phot.png")

from sklearn.metrics import r2_score, mean_squared_error

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"R2 Score: {r2:.4f}")

# ================================
# 2. RESIDUAL ANALYSIS
# ================================

residual = y_test - y_pred

plt.figure(figsize=(6,4))

plt.scatter(y_test, residual, s=5)

plt.axhline(0, color="red")

plt.xlabel("Spectroscopic Redshift (z_spec)")
plt.ylabel("Residual (z_spec - z_phot)")
plt.title("Residual vs Redshift")

plt.savefig("residual_vs_redshift.png")

plt.show()

# ================================
# 3. DISTRIBUSI ERROR
# ================================

plt.figure(figsize=(6,4))

plt.hist(residual, bins=60)

plt.xlabel("Residual (z_spec - z_phot)")
plt.ylabel("Number of Galaxies")
plt.title("Residual Distribution")

plt.show()


# ================================
# 4. FEATURE IMPORTANCE
# ================================

feature_names = X_train.columns

importance = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:")
print(feature_importance)


plt.figure(figsize=(8,5))

plt.bar(feature_importance["Feature"],
        feature_importance["Importance"])

plt.xticks(rotation=45)

plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")

plt.show()

delta_z = y_test - y_pred

plt.figure(figsize=(6,6))

sc = plt.scatter(
    y_test,
    y_pred,
    c=delta_z,
    cmap="coolwarm",
    s=5
)

plt.colorbar(sc, label="Residual (z_spec - z_phot)")

plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color="black")

plt.xlabel("z_spec")
plt.ylabel("z_phot")
plt.title("Photometric Redshift with Residual Coloring")

plt.show()

print("===== ANALYSIS FINISHED =====")

# =========================================
# HYPERPARAMETER TUNING - RANDOM FOREST
# =========================================

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

# 1. Definisikan model dasar
rf = RandomForestRegressor(random_state=42)

# =========================================
# 2. Definisikan ruang parameter
# =========================================

param_dist = {
    "n_estimators": [100, 200, 300],        # jumlah pohon
    "max_depth": [10, 20, 30, None],        # kedalaman pohon
    "min_samples_split": [2, 5, 10],        # minimum split
    "min_samples_leaf": [1, 2, 4],          # minimum leaf
    "max_features": ["sqrt", "log2"]        # jumlah fitur tiap split
}

# =========================================
# 3. Random Search (lebih ringan dari Grid)
# =========================================

random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=10,              # jumlah kombinasi yang dicoba
    cv=3,                   # cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1               # pakai semua CPU
)

# =========================================
# 4. Training dengan tuning
# =========================================

random_search.fit(X_train, y_train)

print("\n===== HASIL TUNING =====")
print("Best Parameters:", random_search.best_params_)

# =========================================
# 5. Gunakan model terbaik
# =========================================

best_model = random_search.best_estimator_

# Prediksi ulang
y_pred_tuned = best_model.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae = mean_absolute_error(y_test, y_pred_tuned)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
r2 = r2_score(y_test, y_pred_tuned)

print("\n===== EVALUASI SETELAH TUNING =====")
print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# CATATAN: Pastikan nama variabel model Anda benar.
# Jika sebelumnya Anda menamai modelnya 'rf_model' atau 'rf_tuned',
# ubah kata 'model' di bawah ini menjadi nama variabel tersebut.

print("Menghitung prediksi untuk Data Latihan dan Data Ujian...")

# 1. AI melakukan tebakan untuk kedua jenis data
y_pred_train = rf_model.predict(X_train) # Tebakan untuk materi yang sudah dipelajari
y_pred_test = rf_model.predict(X_test)   # Tebakan untuk soal ujian baru

# 2. Menghitung nilai R-Squared (R^2)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

# 3. Membuat Visualisasi (Dua grafik berdampingan)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Batas garis diagonal (disesuaikan dengan batas z < 0.4 atau sesuai data Anda)
batas_min = min(y_test.min(), y_train.min())
batas_max = max(y_test.max(), y_train.max())

# ---- GRAFIK 1: DATA TRAINING (Biru) ----
axes[0].scatter(y_train, y_pred_train, alpha=0.3, color='blue', s=10)
axes[0].plot([batas_min, batas_max], [batas_min, batas_max], 'k--', lw=2) # Garis diagonal ideal
axes[0].set_title(f"DATA TRAINING\n$R^2$ Score: {r2_train:.4f}", fontsize=14, color='darkblue')
axes[0].set_xlabel("Redshift Asli ($z_{spec}$)", fontsize=12)
axes[0].set_ylabel("Prediksi AI ($z_{phot}$)", fontsize=12)
axes[0].grid(True, linestyle=':', alpha=0.6)

# ---- GRAFIK 2: DATA TESTING (Merah) ----
axes[1].scatter(y_test, y_pred_test, alpha=0.3, color='red', s=10)
axes[1].plot([batas_min, batas_max], [batas_min, batas_max], 'k--', lw=2) # Garis diagonal ideal
axes[1].set_title(f"DATA TESTING\n$R^2$ Score: {r2_test:.4f}", fontsize=14, color='darkred')
axes[1].set_xlabel("Redshift Asli ($z_{spec}$)", fontsize=12)
axes[1].set_ylabel("Prediksi AI ($z_{phot}$)", fontsize=12)
axes[1].grid(True, linestyle=':', alpha=0.6)

# Kosmetik tampilan
plt.suptitle("Analisis Performa Model: Deteksi Overfitting vs Underfitting", fontsize=16, y=1.05)
plt.tight_layout()
plt.show()

# 4. Kesimpulan Otomatis (Diagnosis)
print("\n" + "="*50)
print("DIAGNOSIS KESEHATAN MODEL AI:")
print(f"Akurasi di Data Training (R^2) : {r2_train:.4f}")
print(f"Akurasi di Data Testing (R^2)  : {r2_test:.4f}")
selisih = r2_train - r2_test
print(f"Selisih Penurunan Akurasi      : {selisih:.4f}")
print("-" * 50)

if r2_train < 0.7:
    print("Kesimpulan: UNDERFITTING.\nModel terlalu sederhana. Belum bisa menangkap pola fisika dari warna galaksi.")
elif selisih > 0.15:
    print("Kesimpulan: OVERFITTING!\nModel hanya menghafal data training, tapi kebingungan saat diberi galaksi baru.")
else:
    print("Kesimpulan: GOOD FIT (IDEAL)!\nModel belajar dengan sangat baik, stabil, dan siap digunakan untuk penelitian sungguhan.")
print("="*50)
