import os
import sys
import glob
import wave
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, DoubleType, IntegerType, StringType
)

# Configuration explicite de l'exécutable Python pour PySpark sous Windows
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

def extract_audio_features(file_path):
    """Extrait la durée et la fréquence d'échantillonnage."""
    try:
        with wave.open(file_path, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            duration = n_frames / float(sample_rate)
            return float(duration), int(sample_rate)
    except Exception:
        return None, None


def main():
    # 1. Initialiser la SparkSession
    print("Initialisation de PySpark...")
    spark = SparkSession.builder \
        .appName("VoiceGender_DataPrep") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")

    output_file = "output_csv.csv"
    path_dataset_pattern = "C:/Users/RETEC/mon_projet_audio/data/*/*.wav"

    # 2. Lister et extraire les informations localement (contournement du bug Windows PySpark Worker)
    files = glob.glob(path_dataset_pattern, recursive=True)
    if not files:
        print("Aucun fichier .wav trouvé dans le dossier indiqué.")
        spark.stop()
        return

    print(f"{len(files)} fichiers trouvés. Extraction des caractéristiques...")

    rows = []
    for file_path in files:
        normalized = file_path.replace("\\", "/")
        parts = normalized.split("/")
        gender = parts[-2] if len(parts) >= 2 else "unknown"

        duration, sample_rate = extract_audio_features(file_path)
        if duration is not None and sample_rate is not None:
            rows.append((normalized, gender, duration, sample_rate))

    if not rows:
        print("Aucune donnée extraite. Vérifiez vos fichiers .wav.")
        spark.stop()
        return

    # 3. Création du DataFrame Spark pour la gestion des données
    print("Création du DataFrame PySpark et sauvegarde...")
    schema = StructType([
        StructField("path", StringType(), False),
        StructField("gender", StringType(), False),
        StructField("duree_sec", DoubleType(), False),
        StructField("frequence_hz", IntegerType(), False)
    ])
    df = spark.createDataFrame(rows, schema=schema)

    # 4. Export via Spark vers Pandas (pour compatibilité avec la suite du pipeline)
    df_pandas = df.toPandas()
    
    # Réorganiser les colonnes
    df_pandas = df_pandas[["gender", "duree_sec", "frequence_hz", "path"]]
    df_pandas.to_csv(output_file, index=False, sep=",")
    
    print(f"\n{len(df_pandas)} fichiers exportés dans : {output_file}\n")
    print(df_pandas.head(20).to_string(index=False))

    spark.stop()


if __name__ == "__main__":
    main()


