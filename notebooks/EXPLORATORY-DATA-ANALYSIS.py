# =========================================
# EXPLORATORY DATA ANALYSIS (EDA)
# =========================================

import matplotlib.pyplot as plt
import seaborn as sns

print("===== EDA START =====")

# ================================
# 1. INFORMASI DATASET
# ================================

print("\nDataset info:")
print(df.info())

print("\nStatistical summary:")
display(df.describe())

print("\nJumlah data:", len(df))


# ================================
# 2. DISTRIBUSI REDSHIFT
# ================================

plt.figure(figsize=(6,4))
plt.hist(df["specz"], bins=60)
plt.xlabel("Spectroscopic Redshift (z)")
plt.ylabel("Number of Galaxies")
plt.title("Redshift Distribution")
plt.show()


# ================================
# 3. DISTRIBUSI MAGNITUDE
# ================================

magnitudes = ["modelMag_u","modelMag_g","modelMag_r","modelMag_i","modelMag_z"]

for mag in magnitudes:
    plt.figure(figsize=(6,4))
    plt.hist(df[mag], bins=60)
    plt.xlabel("Magnitude")
    plt.ylabel("Count")
    plt.title(f"Distribution of {mag}")
    plt.show()


# ================================
# 4. DISTRIBUSI COLOR INDEX
# ================================

colors = ["u_g","g_r","r_i","i_z"]

for c in colors:
    plt.figure(figsize=(6,4))
    plt.hist(df[c], bins=60)
    plt.xlabel("Color Index")
    plt.ylabel("Count")
    plt.title(f"Distribution of {c}")
    plt.show()


# ================================
# 5. HUBUNGAN COLOR vs REDSHIFT
# ================================

for c in colors:
    plt.figure(figsize=(6,5))
    plt.scatter(df[c], df["specz"], s=1)
    plt.xlabel(c)
    plt.ylabel("Spectroscopic Redshift (z)")
    plt.title(f"{c} vs Redshift")
    plt.show()


# ================================
# 6. MAGNITUDE vs REDSHIFT
# ================================

for mag in magnitudes:
    plt.figure(figsize=(6,5))
    plt.scatter(df[mag], df["specz"], s=1)
    plt.xlabel(mag)
    plt.ylabel("Spectroscopic Redshift (z)")
    plt.title(f"{mag} vs Redshift")
    plt.show()


# ================================
# 7. CORRELATION MATRIX
# ================================

plt.figure(figsize=(10,8))
corr = df.corr()

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Matrix")
plt.show()


# ================================
# 8. PAIRPLOT (SAMPLING AGAR RINGAN)
# ================================

print("\nGenerating pairplot (sampled data)...")

sample_df = df.sample(5000)

sns.pairplot(
    sample_df,
    vars=["u_g","g_r","r_i","i_z","specz"],
    corner=True,
    plot_kws={"s":3}
)

plt.show()


print("\n===== EDA FINISHED =====")
