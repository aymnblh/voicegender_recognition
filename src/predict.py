
import sys
import os
import joblib
import pandas as pd
from feature_extraction import extract_features

MODEL_PATH = "models/best_model.pkl"

def predict(audio_path):
    if not os.path.exists(audio_path):
        print(f"❌ Erreur : Le fichier {audio_path} n'existe pas.")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"❌ Erreur : Le modèle n'a pas été trouvé dans {MODEL_PATH}.")
        print("   Veuillez d'abord exécuter : python src/train_model.py")
        return

    # 1. Charger le modèle et le label encoder
    print(f"--- Chargement du modèle ---")
    data_bundle = joblib.load(MODEL_PATH)
    model = data_bundle["model"]
    le = data_bundle["label_encoder"]

    # 2. Extraire les features du fichier cible
    print(f"--- Analyse du fichier : {os.path.basename(audio_path)} ---")
    raw_features = extract_features({"path": audio_path, "gender": "unknown"})
    
    if raw_features is None:
        print("❌ Erreur : Impossible d'extraire les caractéristiques de l'audio.")
        return

    # Convertir en DataFrame (en enlevant les colonnes non-numériques utilisées par le modèle)
    df_features = pd.DataFrame([raw_features]).drop(columns=["gender", "path"])

    # 3. Prédiction
    prediction_idx = model.predict(df_features)[0]
    prob = model.predict_proba(df_features)[0]
    
    label = le.inverse_transform([prediction_idx])[0]
    confidence = prob[prediction_idx] * 100

    print(f"\n🎯 Résultat : {label.upper()}")
    print(f"📈 Confiance : {confidence:.2f}%")
    
    # Détails des probabilités
    for i, class_name in enumerate(le.classes_):
        print(f"   - {class_name}: {prob[i]*100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py <chemin_du_fichier_wav>")
    else:
        predict(sys.argv[1])
