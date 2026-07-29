import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- LANGKAH PENTING: DEFINISIKAN VARIABEL YANG HILANG ---

# 1. Definisikan y_test_class dari y_test (Data Asli)
threshold = 0.3
y_test_class = np.where(y_test <= threshold, 'Near', 'Mid')

# 2. Definisikan y_pred_class dari hasil prediksi model
# Ganti 'y_pred' dengan nama variabel hasil prediksi model XGBoost/FCN Anda
y_pred_class = np.where(y_pred <= threshold, 'Near', 'Mid')

# -------------------------------------------------------

# # 3. Menghitung Confusion Matrix
cm = confusion_matrix(y_test_class, y_pred_class)
labels = sorted(np.unique(y_test_class))

# # 4. Membuat Plot Heatmap
plt.figure(figsize=(8, 6))
sns.set_theme(style="white")
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)

plt.title('Confusion Matrix: Klasifikasi Jarak Galaksi', fontsize=14, fontweight='bold')
plt.xlabel('Prediksi Model', fontsize=12)
plt.ylabel('Data Asli (Actual)', fontsize=12)
plt.savefig("hasil_klasifikasi_galaksi.png")
plt.show()
