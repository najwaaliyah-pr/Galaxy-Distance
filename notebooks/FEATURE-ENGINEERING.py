# ============================================
# FEATURE ENGINEERING - PHOTOMETRIC REDSHIFT
# ============================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load Dataset
df = pd.read_csv('sdss_dr16_galaxy_100k.csv')

print("Jumlah data awal:", df.shape)
print(df.head())

# ============================================
# Data Cleaning
# ============================================

df = df.dropna()
print("Jumlah data setelah hapus NA:", df.shape)

# ============================================
# Feature Engineering (Color Index)
# ============================================

df['u_g'] = df['modelMag_u'] - df['modelMag_g']
df['g_r'] = df['modelMag_g'] - df['modelMag_r']
df['r_i'] = df['modelMag_r'] - df['modelMag_i']
df['i_z'] = df['modelMag_i'] - df['modelMag_z']

# ============================================
# Log Transform
# ============================================

df['log_g'] = np.log1p(df['modelMag_g'])
df['log_r'] = np.log1p(df['modelMag_r'])
df['log_i'] = np.log1p(df['modelMag_i'])

# ============================================
# Outlier Removal
# ============================================

df = df[(df['specz'] > 0) & (df['specz'] < 1)]
df = df[(df['modelMag_g'] > 10) & (df['modelMag_g'] < 25)]

print("Jumlah data setelah filtering:", df.shape)

# ============================================
# Feature Selection
# ============================================

features = [
    'modelMag_u',
    'modelMag_g',
    'modelMag_r',
    'modelMag_i',
    'modelMag_z',
    'u_g',
    'g_r',
    'r_i',
    'i_z',
    'log_g',
    'log_r',
    'log_i'
]

X = df[features]
y = df['specz']

# ============================================
# Feature Scaling
# ============================================

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(X_scaled, columns=features)

# ============================================
# Output Dataset
# ============================================

print("\nShape fitur:", X_scaled.shape)
print("Shape target:", y.shape)

print("\nContoh fitur setelah scaling:")
print(X_scaled.head())

print("\nTarget (redshift):")
print(y.head())
