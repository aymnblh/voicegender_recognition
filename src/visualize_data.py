"""
visualize_data.py
-----------------
Analyse exploratoire des données (EDA) sur les caractéristiques extraites.
Ce script lit `features.csv` via PySpark, puis génère des visualisations 
statistiques (distributions, boxplots, corrélations) sauvegardées dans 
`models/visualizations/`.
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession

# Configuration explicite de l'exécutable Python pour PySpark sous Windows
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

FEATURES_CSV = "features.csv"
VISUALIZATIONS_DIR = "models/visualizations"


def create_visualizations(df_pandas):
    """Génère et sauvegarde toutes les visualisations pertinentes."""
    print("Génération des graphiques...")

    # Palette de couleurs personnalisée pour respecter le 'design aesthetics'
    colors = {"male": "#2E86C1", "women": "#E74C3C"}

    # 1. Distribution de la durée par genre (KDE Plot)
    if "duree_sec" in df_pandas.columns:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df_pandas, x="duree_sec", hue="gender", fill=True, 
                    palette=colors, alpha=0.5, linewidth=2)
        plt.title("Distribution de la durée des fichiers audio par genre", fontsize=16)
        plt.xlabel("Durée (secondes)", fontsize=12)
        plt.ylabel("Densité", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{VISUALIZATIONS_DIR}/distribution_duree.png")
        plt.close()

    # 2. Boxplot : Fréquence fondamentale (F0 - Pitch)
    # C'est souvent le différenciateur le plus fort.
    if "f0_mean" in df_pandas.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df_pandas, x="gender", y="f0_mean", palette=colors, 
                    width=0.5, fliersize=2)
        plt.title("Répartition de la Fréquence Fondamentale (Pitch) par genre", fontsize=16)
        plt.xlabel("Genre", fontsize=12)
        plt.ylabel("Fréquence F0 moyenne (Hz)", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{VISUALIZATIONS_DIR}/boxplot_f0.png")
        plt.close()

    # 3. Boxplot : Spectral Centroid (Luminance du son)
    if "spectral_centroid_mean" in df_pandas.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df_pandas, x="gender", y="spectral_centroid_mean", 
                    palette=colors, width=0.5, fliersize=2)
        plt.title("Spectral Centroid (Luminance apparente) par genre", fontsize=16)
        plt.xlabel("Genre", fontsize=12)
        plt.ylabel("Spectral Centroid (Moyenne)", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{VISUALIZATIONS_DIR}/boxplot_spectral_centroid.png")
        plt.close()

    # 4. Matrice de corrélation
    print("Génération de la matrice de corrélation (15 premières variables numériques)...")
    # Sélectionner uniquement les caractéristiques numériques
    numeric_cols = df_pandas.select_dtypes(include=['float64', 'int64']).columns
    
    # Prendre les 15 premières si on en a trop pour que ça reste lisible
    cols_to_plot = numeric_cols[:15] if len(numeric_cols) > 15 else numeric_cols
    
    if len(cols_to_plot) > 0:
        plt.figure(figsize=(12, 10))
        corr_matrix = df_pandas[cols_to_plot].corr()
        
        # Mask pour ne garder que la moitié inférieure du triangle
        import numpy as np
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, cmap="coolwarm", annot=True, 
                    fmt=".2f", vmin=-1, vmax=1, square=True, 
                    cbar_kws={"shrink": .8}, linewidths=.5, cbar=True)
        
        plt.title("Matrice de Corrélation des Caractéristiques Pincipales", fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(f"{VISUALIZATIONS_DIR}/matrice_correlation.png", dpi=150)
        plt.close()

    print(f"Visualisations sauvegardées dans : {VISUALIZATIONS_DIR}/")


def main():
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    
    print("Initialisation de PySpark pour l'Analyse des Données...")
    spark = SparkSession.builder \
        .appName("VoiceGender_EDA") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")

    if not os.path.exists(FEATURES_CSV):
        print(f" Fichier {FEATURES_CSV} introuvable. Exécutez d'abord src/feature_extraction.py")
        spark.stop()
        return

    print(f"Lecture du fichier : {FEATURES_CSV} via PySpark...")
    df_spark = spark.read.csv(FEATURES_CSV, header=True, inferSchema=True)
    
    # Filtrer les genres si nécessaire (garantir la propreté des graphes)
    df_spark_filtered = df_spark.filter(df_spark.gender.isin("male", "women"))
    
    print("Conversion vers Pandas pour le traçage...")
    # Pour Matplotlib/Seaborn, on doit ramener les données en mémoire (Pandas)
    df_pandas = df_spark_filtered.toPandas()

    if df_pandas.empty:
        print(" Aucune donnée à visualiser.")
        spark.stop()
        return

    print(f"Données chargées : {len(df_pandas)} échantillons.")
    create_visualizations(df_pandas)

    spark.stop()

if __name__ == "__main__":
    main()
