"""
feature_extraction.py
extraction des features audio avec librosa
sauvegarde en parquet pour l'entrainement
optimise pour big data avec udf spark
"""

import os
import sys
import numpy as np
import librosa
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType
)

os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

N_MFCC     = 13
INPUT_PARQUET  = "/output/files.parquet"
OUTPUT_PARQUET = "/output/features.parquet"

def extract_features_from_path(file_path: str, gender: str) -> dict:
    """extraire les features audio d'un fichier wav"""
    try:
        # nettoyer les prefixes spark
        clean_path = file_path
        if clean_path.startswith("file://"):
            clean_path = clean_path[7:]
        if clean_path.startswith("file:"):
            clean_path = clean_path[5:]

        if not os.path.exists(clean_path):
            return None

        y, sr = librosa.load(clean_path, sr=None, mono=True)
    except Exception:
        return None

    feats = {}

    # mfcc
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    for i in range(N_MFCC):
        feats[f"mfcc{i+1}_mean"] = float(np.mean(mfcc[i]))
        feats[f"mfcc{i+1}_std"]  = float(np.std(mfcc[i]))

    # zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)
    feats["zcr_mean"] = float(np.mean(zcr))
    feats["zcr_std"]  = float(np.std(zcr))

    # spectral features
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
    feats["spectral_centroid_mean"] = float(np.mean(sc))
    feats["spectral_centroid_std"]  = float(np.std(sc))

    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    feats["spectral_bandwidth_mean"] = float(np.mean(sb))
    feats["spectral_bandwidth_std"]  = float(np.std(sb))

    sro = librosa.feature.spectral_rolloff(y=y, sr=sr)
    feats["spectral_rolloff_mean"] = float(np.mean(sro))
    feats["spectral_rolloff_std"]  = float(np.std(sro))

    # rms energy
    rms = librosa.feature.rms(y=y)
    feats["rms_mean"] = float(np.mean(rms))
    feats["rms_std"]  = float(np.std(rms))

    # chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    for i in range(12):
        feats[f"chroma{i+1}_mean"] = float(np.mean(chroma[i]))

    # f0 frequency
    try:
        f0 = librosa.yin(y, fmin=50, fmax=500)
        f0_voiced = f0[f0 > 0]
        feats["f0_mean"] = float(np.mean(f0_voiced)) if len(f0_voiced) > 0 else 0.0
        feats["f0_std"]  = float(np.std(f0_voiced))  if len(f0_voiced) > 0 else 0.0
    except Exception:
        feats["f0_mean"] = 0.0
        feats["f0_std"]  = 0.0

    return feats

def extract_features_spark(row_dict):
    return extract_features_from_path(row_dict["path"], row_dict["gender"])


def main():
    print("initialisation pyspark")
    spark = SparkSession.builder \
        .appName("VoiceGender_FeatureExtraction") \
        .master("local[*]") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .config("spark.python.worker.memory", "1g") \
        .config("spark.python.worker.reuse", "true") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    print(f"lecture: {INPUT_PARQUET}")
    df_input = spark.read.parquet(INPUT_PARQUET)

    file_count = df_input.count()
    print(f"{file_count} fichiers trouves")

    print("\npremiers chemins:")
    df_input.select("path", "gender").show(3, truncate=False)

    if file_count == 0:
        print("aucune donnee")
        spark.stop()
        return

    # schema des features en sortie
    features_schema = StructType([
        StructField("mfcc1_mean", DoubleType()), StructField("mfcc1_std", DoubleType()),
        StructField("mfcc2_mean", DoubleType()), StructField("mfcc2_std", DoubleType()),
        StructField("mfcc3_mean", DoubleType()), StructField("mfcc3_std", DoubleType()),
        StructField("mfcc4_mean", DoubleType()), StructField("mfcc4_std", DoubleType()),
        StructField("mfcc5_mean", DoubleType()), StructField("mfcc5_std", DoubleType()),
        StructField("mfcc6_mean", DoubleType()), StructField("mfcc6_std", DoubleType()),
        StructField("mfcc7_mean", DoubleType()), StructField("mfcc7_std", DoubleType()),
        StructField("mfcc8_mean", DoubleType()), StructField("mfcc8_std", DoubleType()),
        StructField("mfcc9_mean", DoubleType()), StructField("mfcc9_std", DoubleType()),
        StructField("mfcc10_mean", DoubleType()), StructField("mfcc10_std", DoubleType()),
        StructField("mfcc11_mean", DoubleType()), StructField("mfcc11_std", DoubleType()),
        StructField("mfcc12_mean", DoubleType()), StructField("mfcc12_std", DoubleType()),
        StructField("mfcc13_mean", DoubleType()), StructField("mfcc13_std", DoubleType()),
        StructField("zcr_mean", DoubleType()), StructField("zcr_std", DoubleType()),
        StructField("spectral_centroid_mean", DoubleType()), StructField("spectral_centroid_std", DoubleType()),
        StructField("spectral_bandwidth_mean", DoubleType()), StructField("spectral_bandwidth_std", DoubleType()),
        StructField("spectral_rolloff_mean", DoubleType()), StructField("spectral_rolloff_std", DoubleType()),
        StructField("rms_mean", DoubleType()), StructField("rms_std", DoubleType()),
        StructField("chroma1_mean", DoubleType()), StructField("chroma2_mean", DoubleType()),
        StructField("chroma3_mean", DoubleType()), StructField("chroma4_mean", DoubleType()),
        StructField("chroma5_mean", DoubleType()), StructField("chroma6_mean", DoubleType()),
        StructField("chroma7_mean", DoubleType()), StructField("chroma8_mean", DoubleType()),
        StructField("chroma9_mean", DoubleType()), StructField("chroma10_mean", DoubleType()),
        StructField("chroma11_mean", DoubleType()), StructField("chroma12_mean", DoubleType()),
        StructField("f0_mean", DoubleType()), StructField("f0_std", DoubleType()),
    ])

    print("extraction des features")

    # extraire les features
    def extract_udf(path, gender):
        try:
            if path is None or gender is None:
                return None

            result = extract_features_from_path(path, gender)
            if result is None:
                return None

            return tuple(result.get(field.name, 0.0) for field in features_schema.fields)

        except Exception:
            return None

    extract_features_udf = F.udf(extract_udf, features_schema)

    df_with_features = df_input.select(
        F.col("gender"),
        F.col("path"),
        extract_features_udf(F.col("path"), F.col("gender")).alias("features")
    ).filter(F.col("features").isNotNull())

    # déballer la struct en colonnes
    df_features = df_with_features.select(
        F.col("gender"),
        F.col("path"),
        *[F.col("features")[field.name].alias(field.name)
          for field in features_schema.fields]
    )

    result_count = df_features.count()
    print(f"\n{result_count} fichiers traites")
    print(f"{len(features_schema.fields)} features par fichier")

    print("sauvegarde parquet")
    df_features.write \
        .mode("overwrite") \
        .option("compression", "snappy") \
        .parquet(OUTPUT_PARQUET)

    print("apercu (5 colonnes):")
    feature_cols = [c for c in df_features.columns if c not in ["path", "gender"]][:5]
    df_features.select(["gender"] + feature_cols).show(5, truncate=False)

    print(f"fichiers sauvegarde: {OUTPUT_PARQUET}")

    spark.stop()

if __name__ == "__main__":
    main()

