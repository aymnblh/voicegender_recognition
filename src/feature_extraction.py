import os
import numpy as np
import pandas as pd
import librosa
from joblib import Parallel, delayed
from tqdm import tqdm



N_MFCC    = 13
N_JOBS    = -1          
MAX_FILES = None        
INPUT_CSV  = "output_csv.csv"
OUTPUT_CSV = "features.csv"


def extract_features(row) -> dict | None:
    """Extrait les features audio d'un fichier WAV. Retourne None si erreur."""
    file_path = row["path"]
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
    except Exception:
        return None

    feats = {"gender": row["gender"], "path": file_path}

    # MFCC
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
    df_meta = pd.read_csv(INPUT_CSV)
    # Exclure les fichiers "raw"
    df_meta = df_meta[df_meta["gender"].isin(["male", "women"])].reset_index(drop=True)

    if MAX_FILES is not None:
        df_meta = df_meta.head(MAX_FILES)

    nb = len(df_meta)
    print(f"Extraction parallèle (N_JOBS={N_JOBS}) sur {nb} fichiers...")

    rows_input = [row for _, row in df_meta.iterrows()]

    results = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(extract_features)(row)
        for row in tqdm(rows_input, unit="fichier")
    )

    rows = [r for r in results if r is not None]
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ {len(df_out)} fichiers traités → {OUTPUT_CSV}")
    print(f"   {df_out.shape[1] - 2} features extraites par fichier\n")
    print(df_out.drop(columns=["path"]).head(5).to_string(index=False))


if __name__ == "__main__":
    main()
