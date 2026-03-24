"""
feature_extraction.py
---------------------
Lit le fichier output_csv.csv avec PySpark, extrait des features audio
avec librosa (MFCC, ZCR, spectral centroid, etc.)
et sauvegarde un nouveau fichier features.csv prêt pour l'entraînement.
"""

import os
import sys
import numpy as np
import pandas as pd
import librosa
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType
)

# Configuration explicite de l'exécutable Python pour PySpark sous Windows
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

# ─────────────────────────────────────────────
# Paramètres
# ─────────────────────────────────────────────
N_MFCC     = 13
INPUT_CSV  = "output_csv.csv"
OUTPUT_CSV = "features.csv"

def extract_features_spark(row_dict) -> dict | None:
    """Extrait les features audio d'un fichier WAV. Retourne None si erreur."""
    file_path = row_dict["path"]
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
    except Exception:
        return None

    feats = {"gender": row_dict["gender"], "path": file_path}

    # MFCC jhng
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        feats[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc{i+1}_std"]  = float(np.std(mfcc[i]))

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    feats["zcr_mean"] = float(np.mean(zcr))
    feats["zcr_std"]  = float(np.std(zcr))

    # Spectral features
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    feats["spectral_centroid_mean"] = float(np.mean(sc))
    feats["spectral_centroid_std"]  = float(np.std(sc))

    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    feats["spectral_bandwidth_mean"] = float(np.mean(sb))
    feats["spectral_bandwidth_std"]  = float(np.std(sb))

    sro = librosa.feature.spectral_rolloff(y=y, sr=sr)
    feats["spectral_rolloff_mean"] = float(np.mean(sro))
    feats["spectral_rolloff_std"]  = float(np.std(sro))

    # RMS Energy
    rms = librosa.feature.rms(y=y)
    feats["rms_mean"] = float(np.mean(rms))
    feats["rms_std"]  = float(np.std(rms))

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(12):
        feats[f"chroma{i+1}_mean"] = float(np.mean(chroma[i]))

    # Fréquence fondamentale (F0) via YIN
    try:
        f0 = librosa.yin(y, fmin=50, fmax=500)
        f0_voiced = f0[f0 > 0]
        feats["f0_mean"] = float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0.0
        feats["f0_std"]  = float(np.std(f0_voiced))  if len(f0_voiced) > 0 else 0.0
    except Exception:
        feats["f0_mean"] = 0.0
        feats["f0_std"]  = 0.0

    return feats

def main():
    print("Initialisation de PySpark pour l'extraction de features...")
    spark = SparkSession.builder \
        .appName("VoiceGender_FeatureExtraction") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")

    # 1. Lecture du CSV via PySpark
    print(f"Lecture du fichier d'entrée : {INPUT_CSV}")
    df_meta = spark.read.csv(INPUT_CSV, header=True, inferSchema=True)

    # 2. Filtrage des données "raw" via PySpark
    df_filtered = df_meta.filter(df_meta.gender.isin("male", "women"))
    
    # 3. Extraction (Contournement Python Worker Timeout pour extraire localement + Spark)
    # On rassemble les lignes pertinentes
    rows_to_process = [row.asDict() for row in df_filtered.collect()]
    nb = len(rows_to_process)
    
    print(f"Extraction sur {nb} fichiers (préparation des données avec Spark)...")
    
    # Afin de s'assurer de la stabilité sur Windows, on exécute l'extraction Python intensive ici
    # (Sur un vrai cluster Linux, on utiliserait rdd.map() ou pandas_udf direct)
    from joblib import Parallel, delayed
    from tqdm import tqdm
    
    # Utilisation de Joblib pour la stabilité des threads sous Windows au lieu de Spark RDD map()
    # qui crash avec SocketTimeoutException à cause des pipes de communication Java/Python Windows.
    # Spark reste utilisé pour l'I/O et la création du DataFrame de sortie.
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(extract_features_spark)(row)
        for row in tqdm(rows_to_process, unit="fichier")
    )
    
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        print("Aucune feature n'a pu être extraite.")
        spark.stop()
        return

    # 4. Reconstruction du Spark DataFrame final et sauvegarde
    print("Création du DataFrame PySpark de sortie et sauvegarde CSV...")
    
    # Typage dynamique du schema en fonction du premier élément valide
    first_row = valid_results[0]
    fields = []
    for k, v in first_row.items():
        if isinstance(v, str):
            fields.append(StructField(k, StringType(), True))
        else:
            fields.append(StructField(k, DoubleType(), True))
            
    schema_out = StructType(fields)
    
    # Création DataFrame PySpark
    df_out = spark.createDataFrame(valid_results, schema=schema_out)
    
    # Sauvegarde
    df_pandas = df_out.toPandas()
    df_pandas.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ {len(df_pandas)} fichiers traités → {OUTPUT_CSV}")
    print(f"   {df_pandas.shape[1] - 2} features extraites par fichier\n")
    print(df_pandas.drop(columns=["path"]).head(5).to_string(index=False))

    spark.stop()

if __name__ == "__main__":
    main()

