import os
import sys
import wave

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (
    StructType, StructField, DoubleType, IntegerType, StringType
)

# Configuration explicite de l'exécutable Python pour PySpark (compatible Docker et Windows)
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"


def extract_wav_files(test_mode=True, test_limit=1000):
    data_path = "/data/*/*.wav"
    output_file = "/output/files.parquet"

    print("initialisation pyspark")
    print(f"  chemin: {data_path}")
    print(f"  sortie: {output_file}")

    spark = SparkSession.builder \
        .appName("VoiceGender_DataPrep") \
        .master("local[*]") \
        .config("spark.driver.host", "127.0.0.1") \
        .config("spark.driver.bindAddress", "127.0.0.1") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    def extract_audio_metadata(file_path):
        """extraire duration et sample_rate"""
        if file_path is None or file_path == "":
            return None

        try:
            clean_path = file_path
            if clean_path.startswith("file://"):
                clean_path = clean_path[7:]
            if clean_path.startswith("file:"):
                clean_path = clean_path[5:]

            if not os.path.exists(clean_path):
                return None

            with wave.open(clean_path, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                duration = n_frames / float(sample_rate)
            return (float(duration), int(sample_rate))
        except Exception as e:
            return None

    extract_udf = F.udf(
        extract_audio_metadata,
        StructType([
            StructField("duration", DoubleType()),
            StructField("sample_rate", IntegerType())
        ])
    )

    print("lecture des fichiers .wav")
    df_files = spark.read.format("binaryFile").load(data_path)

    file_count = df_files.count()
    print(f"{file_count} fichiers trouves")

    if test_mode and file_count > test_limit:
        print(f"mode test: {test_limit} fichiers (sur {file_count})")
        df_files = df_files.limit(test_limit)
        file_count = test_limit

    if file_count == 0:
        print(f"aucun fichier trouve. path: {data_path}")
        spark.stop()
        return

    print("premiers chemins:")
    df_files.select(F.col("path")).show(5, truncate=False)

    df_files = df_files.select(
        F.regexp_replace(F.col("path"), "^file://", "").alias("path")
    )

    print("chemins normalises:")
    df_files.select(F.col("path")).show(5, truncate=False)

    df_with_gender = df_files.select(
        F.col("path"),
        F.regexp_extract(F.col("path"), r"/([^/]+)/[^/]+\.wav$", 1).alias("gender")
    ).filter(F.col("gender").isNotNull() & (F.col("gender") != ""))

    gender_count = df_with_gender.count()
    print(f"{gender_count} fichiers avec genre")
    if gender_count > 0:
        print("genres trouves:")
        df_with_gender.select(F.col("path"), F.col("gender")).show(5, truncate=False)

    print("extraction des metadonnees")
    df_features = df_with_gender.select(
        F.col("path"),
        F.col("gender"),
        extract_udf(F.col("path")).alias("audio_features")
    ).select(
        F.col("path"),
        F.col("gender"),
        F.col("audio_features.duration").alias("duree_sec"),
        F.col("audio_features.sample_rate").alias("frequence_hz")
    )

    df_clean = df_features.filter(F.col("duree_sec").isNotNull())

    df_final = df_clean.select(
        F.col("gender"),
        F.col("duree_sec"),
        F.col("frequence_hz"),
        F.col("path")
    )

    row_count = df_final.count()

    if row_count == 0:
        print("aucune donnee extraite")
        spark.stop()
        return

    print(f"sauvegarde: {row_count} fichiers")

    df_final.write \
        .mode("overwrite") \
        .option("compression", "snappy") \
        .parquet(output_file)

    print("apercu:")
    df_final.show(20, truncate=False)

    print(f"fichiers sauvegardé : {output_file}")

    spark.stop()

if __name__ == "__main__":
    extract_wav_files()


