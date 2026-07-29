# ================================
# DATA AUDIT PIPELINE
# SDSS DR16 - Photometric Redshift
# ================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================================
# 1. BASIC INFORMATION
# ================================
print("===== BASIC INFO =====")
print(df.info())

print("\n===== DATASET SHAPE =====")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# ================================
# 2. DESCRIPTIVE STATISTICS
# ================================
print("\n===== DESCRIPTIVE STATISTICS =====")
display(df.describe())

# ================================
# 3. MISSING VALUE CHECK
# ================================
print("\n===== MISSING VALUES =====")
display(df.isnull().sum())

# ================================
# 4. REDSHIFT DISTRIBUTION
# ================================
plt.figure(figsize=(7,5))
plt.hist(df['specz'], bins=100)
plt.xlabel("Spectroscopic Redshift (z)")
plt.ylabel("Count")
plt.title("Redshift Distribution")
plt.show()

# ================================
# 5. MAGNITUDE DISTRIBUTIONS
# ================================
bands = ['modelMag_u','modelMag_g','modelMag_r','modelMag_i','modelMag_z']

for b in bands:
    plt.figure(figsize=(7,5))
    plt.hist(df[b], bins=100)
    plt.xlabel(b)
    plt.ylabel("Count")
    plt.title(f"Distribution of {b}")
    plt.show()

# ================================
# 6. OUTLIER & PHYSICAL CHECK
# ================================
print("\n===== NEGATIVE REDSHIFT COUNT =====")
print((df['specz'] < 0).sum())

print("\n===== EXTREME MAGNITUDE COUNT =====")
for b in bands:
    count = ((df[b] < 10) | (df[b] > 30)).sum()
    print(f"{b}: {count}")

# ================================
# 7. COLOR VS REDSHIFT RELATION
# ================================
colors = ['u_g','g_r','r_i','i_z']

for c in colors:
    plt.figure(figsize=(7,5))
    plt.scatter(df[c], df['specz'], s=1, alpha=0.3)
    plt.xlabel(c)
    plt.ylabel("z")
    plt.title(f"{c} vs Redshift")
    plt.show()

# ================================
# 8. COLOR-COLOR DIAGRAM
# ================================
plt.figure(figsize=(7,5))
plt.scatter(df['u_g'], df['g_r'], s=1, alpha=0.3)
plt.xlabel("u-g")
plt.ylabel("g-r")
plt.title("Color–Color Diagram")
plt.show()

print("\n===== DATA AUDIT FINISHED =====")

print("\n===== DATA CLEANING START =====")

# simpan jumlah awal
initial_size = len(df)

# =====================================
# 1. Remove missing values
# =====================================
df = df.dropna()

print("After removing NaN:", len(df))

# =====================================
# 2. Remove impossible magnitude values
# SDSS magnitude biasanya berada di range:
# -5 < mag < 30
# =====================================

mag_columns = ['modelMag_u','modelMag_g','modelMag_r','modelMag_i','modelMag_z']

for col in mag_columns:
    df = df[(df[col] > -5) & (df[col] < 30)]

print("After magnitude filtering:", len(df))

# =====================================
# 3. Remove unrealistic color values
# color index biasanya dalam range -2 sampai 5
# =====================================

color_columns = ['u_g','g_r','r_i','i_z']

for col in color_columns:
    df = df[(df[col] > -2) & (df[col] < 5)]

print("After color filtering:", len(df))

# =====================================
# 4. Remove invalid redshift
# =====================================

df = df[(df['specz'] > 0) & (df['specz'] < 1)]

print("After redshift filtering:", len(df))

# =====================================
# 5. Remove duplicate rows
# =====================================

df = df.drop_duplicates()

print("After removing duplicates:", len(df))

# =====================================
# 6. Reset index
# =====================================

df = df.reset_index(drop=True)

# =====================================
# SUMMARY
# =====================================

final_size = len(df)

print("\n===== CLEANING SUMMARY =====")
print("Initial data:", initial_size)
print("Final data:", final_size)
print("Removed:", initial_size - final_size)

print("\n===== DATA CLEANING FINISHED =====")
