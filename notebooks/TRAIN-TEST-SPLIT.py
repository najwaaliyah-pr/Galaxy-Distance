# ============================================
# TRAIN - TEST SPLIT
# ============================================

from sklearn.model_selection import train_test_split

# Membagi dataset menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,   # fitur
    y,          # target redshift
    test_size=0.2,   # 20% data untuk testing
    random_state=42  # agar hasil pembagian selalu sama
)

# ============================================
# CEK DIMENSI DATA
# ============================================

print("Jumlah data training:", X_train.shape)
print("Jumlah data testing:", X_test.shape)

print("\nTarget training:", y_train.shape)
print("Target testing:", y_test.shape)
