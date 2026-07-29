# ===============================
# QUERY DATA SDSS DR16 (SDSS-IV)
# ===============================

!pip install astroquery -q

# Import library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astroquery.sdss import SDSS

print("Library siap!")

# ===============================
# SQL QUERY SDSS (REVISI)
# ===============================

# Menggunakan modelMag karena lebih stabil untuk profil galaksi
# Menggunakan alias s.z agar tidak tertukar dengan filter z fotometri
query = """
SELECT TOP 100000
    p.modelMag_u, p.modelMag_g, p.modelMag_r, p.modelMag_i, p.modelMag_z,
    s.z AS specz
FROM PhotoObj AS p
JOIN SpecObj AS s ON s.bestobjid = p.objid
WHERE
    s.class = 'GALAXY'
    AND s.z BETWEEN 0.01 AND 0.7
    AND s.zWarning = 0
    AND p.clean = 1
    AND p.modelMag_r < 20
"""

print("Menjalankan query SDSS DR16...")
# Menambahkan data_release=16 secara eksplisit
data = SDSS.query_sql(query, data_release=16)

if data is not None:
    # Convert ke pandas DataFrame
    df = data.to_pandas()
    print("Query berhasil!")
    print("Jumlah data awal:", df.shape)

    # ===============================
    # FEATURE ENGINEERING: WARNA
    # ===============================
    # Menghitung color index (fitur wajib untuk Photo-z)
    df["u_g"] = df["modelMag_u"] - df["modelMag_g"]
    df["g_r"] = df["modelMag_g"] - df["modelMag_r"]
    df["r_i"] = df["modelMag_r"] - df["modelMag_i"]
    df["i_z"] = df["modelMag_i"] - df["modelMag_z"]

    # Menghapus baris yang mungkin memiliki nilai null atau anomali
    df.replace(-9999, np.nan, inplace=True)
    df.dropna(inplace=True)

    print("Jumlah data setelah dibersihkan:", df.shape)

    # ===============================
    # CEK DATA & SIMPAN
    # ===============================
    display(df.head())

    df.to_csv("sdss_dr16_galaxy_100k.csv", index=False)
    print("\nDataset disimpan sebagai: sdss_dr16_galaxy_100k.csv")
else:
    print("Gagal mengambil data. Server mungkin sibuk atau query salah.")
