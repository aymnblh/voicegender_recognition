"""
visualize_data.py
analyse exploratoire des donnees (eda)
visualisations des features extraites
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FEATURES_PARQUET = "output/features.parquet"
VISUALIZATIONS_DIR = "models/visualizations"


def create_visualizations(df_pandas):
    """generer et sauvegarder les visualisations"""
    print("generation des graphiques")

    colors = {"male": "#2E86C1", "female": "#E74C3C"}

    # distribution duree par genre
    if "duree_sec" in df_pandas.columns:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df_pandas, x="duree_sec", hue="gender", fill=True, 
                    palette=colors, alpha=0.5, linewidth=2)
        plt.title("distribution duree par genre", fontsize=16)
        plt.xlabel("duree (secondes)", fontsize=12)
        plt.ylabel("densite", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{VISUALIZATIONS_DIR}/distribution_duree.png")
        plt.close()

    # boxplot f0 (pitch)
    if "f0_mean" in df_pandas.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df_pandas, x="gender", y="f0_mean", palette=colors, 
                    width=0.5, fliersize=2)
        plt.title("frequence fondamentale (pitch) par genre", fontsize=16)
        plt.xlabel("genre", fontsize=12)
        plt.ylabel("f0 moyenne (hz)", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{VISUALIZATIONS_DIR}/boxplot_f0.png")
        plt.close()

    # boxplot spectral centroid
    if "spectral_centroid_mean" in df_pandas.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df_pandas, x="gender", y="spectral_centroid_mean", 
                    palette=colors, width=0.5, fliersize=2)
        plt.title("spectral centroid par genre", fontsize=16)
        plt.xlabel("genre", fontsize=12)
        plt.ylabel("spectral centroid (moyenne)", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{VISUALIZATIONS_DIR}/boxplot_spectral_centroid.png")
        plt.close()

    # matrice correlation
    print("matrice correlation (15 premieres variables)")
    numeric_cols = df_pandas.select_dtypes(include=['float64', 'int64']).columns
    
    cols_to_plot = numeric_cols[:15] if len(numeric_cols) > 15 else numeric_cols
    
    if len(cols_to_plot) > 0:
        plt.figure(figsize=(12, 10))
        corr_matrix = df_pandas[cols_to_plot].corr()
        
        import numpy as np
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, mask=mask, cmap="coolwarm", annot=True, 
                    fmt=".2f", vmin=-1, vmax=1, square=True, 
                    cbar_kws={"shrink": .8}, linewidths=.5, cbar=True)
        
        plt.title("matrice correlation", fontsize=16, pad=20)
        plt.tight_layout()
        plt.savefig(f"{VISUALIZATIONS_DIR}/matrice_correlation.png", dpi=150)
        plt.close()

    print(f"visualisations: {VISUALIZATIONS_DIR}/")


def main():
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    
    print("chargement des donnees")

    if not os.path.exists(FEATURES_PARQUET):
        print(f"erreur: {FEATURES_PARQUET} non trouve")
        return

    print(f"lecture: {FEATURES_PARQUET}")
    df_pandas = pd.read_parquet(FEATURES_PARQUET)

    # filtrer les genres valides
    df_pandas = df_pandas[df_pandas["gender"].isin(["male", "female"])]

    if df_pandas.empty:
        print("aucune donnee")
        return

    print(f"donnees chargees: {len(df_pandas)} echantillons")
    create_visualizations(df_pandas)


if __name__ == "__main__":
    main()
