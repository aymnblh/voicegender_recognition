import os
import io
import wave
import uuid
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
from src.feature_extraction import extract_features_spark
import soundfile as sf
import subprocess

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Limite à 16MB

MODEL_PATH = "models/best_model.pkl"
TEMP_DIR = "temp"

# S'assurer que le dossier temporaire existe
os.makedirs(TEMP_DIR, exist_ok=True)

# Charger le modèle globalement au démarrage
print(f"--- Chargement du modèle depuis {MODEL_PATH} ---")
try:
    data_bundle = joblib.load(MODEL_PATH)
    model = data_bundle["model"]
    le = data_bundle["label_encoder"]
    print("✅ Modèle chargé avec succès.")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle : {e}")
    model = None
    le = None

@app.route('/')
def home():
    """Sert la page principale HTML."""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_audio():
    """API pour recevoir l'audio et renvoyer la prédiction."""
    if model is None:
        return jsonify({"error": "Modèle non chargé sur le serveur. Entraînez le modèle d'abord."}), 500

    if 'audio' not in request.files:
        return jsonify({"error": "Aucun fichier audio n'a été fourni"}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "Fichier audio vide"}), 400

    try:
        # 1. Sauvegarder l'audio temporairement (Il est déjà converti en WAV par le navigateur)
        file_id = str(uuid.uuid4())
        temp_wav_path = os.path.join(TEMP_DIR, f"{file_id}.wav")
        
        audio_file.save(temp_wav_path)
        print(f"--- Fichier WAV pur reçu du navigateur. Taille : {os.path.getsize(temp_wav_path)} bytes ---")
        
        # 2. Extraction des caractéristiques via la fonction existante
        row_dict = {"path": temp_wav_path, "gender": "unknown"}
        raw_features = extract_features_spark(row_dict)

        if raw_features is None:
            return jsonify({"error": "Impossible d'extraire les caractéristiques audio. Audio trop silencieux ou corrompu."}), 400

        # Formater pour le modèle
        df_features = pd.DataFrame([raw_features]).drop(columns=["gender", "path"])

        # 4. Prédiction
        prediction_idx = model.predict(df_features)[0]
        prob = model.predict_proba(df_features)[0]
        
        label = le.inverse_transform([prediction_idx])[0]
        confidence = prob[prediction_idx] * 100
        
        # Détails complets pour toutes les classes
        details = {class_name: float(prob[i]*100) for i, class_name in enumerate(le.classes_)}

        # 5. Nettoyer les fichiers temporaires
        try:
            if os.path.exists(temp_webm_path): os.remove(temp_webm_path)
            if os.path.exists(temp_wav_path): os.remove(temp_wav_path)
        except Exception as e:
            print(f"Warning: Impossible de supprimer le fichier temp {e}")

        # 6. Renvoyer la réponse
        return jsonify({
            "success": True,
            "prediction": label,
            "confidence": round(confidence, 2),
            "details": details
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Mode debug activé pour faciliter le dev local, écoutant sur le port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
