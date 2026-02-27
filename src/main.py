import os
import sys
import glob
import wave
import pandas as pd


def extract_audio_features(file_path):
    """Extrait la durée et la fréquence d'échantillonnage en lisant uniquement l'en-tête WAV."""
    try:
        with wave.open(file_path, 'rb') as wav_file:
            sample_rate = wav_file.getframerate()
            n_frames = wav_file.getnframes()
            duration = n_frames / float(sample_rate)
            return float(duration), int(sample_rate)
    except Exception:
        return None, None


def main():
    output_file = "output_csv.csv"
    path_dataset_pattern = "C:/Users/RETEC/mon_projet_audio/data/*/*.wav"

    # 3 Lecture et labellisation via Python pur (pas de UDF Spark sur Windows)
    files = glob.glob(path_dataset_pattern, recursive=True)
    if not files:
        print("Aucun fichier .wav trouvé dans le dossier indiqué.")
        return

    print(f"{len(files)} fichiers trouvés. Extraction des caractéristiques...")

    rows = []
    for file_path in files:
        # Extraire le genre depuis le dossier parent
        normalized = file_path.replace("\\", "/")
        parts = normalized.split("/")
        gender = parts[-2] if len(parts) >= 2 else "unknown"

        duration, sample_rate = extract_audio_features(file_path)
        if duration is not None and sample_rate is not None:
            rows.append({
                "gender": gender,
                "duree_sec": duration,
                "frequence_hz": sample_rate,
                "path": normalized
            })

    if not rows:
        print("Aucune donnée extraite. Vérifiez vos fichiers .wav.")
        return

    # 4 Création du DataFrame Pandas
    df = pd.DataFrame(rows)

    # 5 Export CSV + aperçu
    output_file = "output_csv.csv"
    df.to_csv(output_file, index=False, sep=",")
    print(f"\n{len(df)} fichiers exportés dans : {output_file}\n")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()