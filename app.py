import os
import uuid
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify
from src.feature_extraction import extract_features_spark

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Limite à 16MB

MODEL_PATH = "models/best_model.pkl"
TEMP_DIR = "temp"

# S'assurer que le dossier temporaire existe
os.makedirs(TEMP_DIR, exist_ok=True)

print("chargement du modele")
try:
    if not os.path.exists(MODEL_PATH):
        print(f"erreur: {MODEL_PATH} non trouve")
        model = None
        le = None
        feature_columns = None
    else:
        data_bundle = joblib.load(MODEL_PATH)
        model = data_bundle["model"]
        le = data_bundle["label_encoder"]
        feature_columns = data_bundle.get("feature_columns", None)
        print("modele charge")
        print(f"classes: {le.classes_}")
        if feature_columns:
            print(f"features attendues: {len(feature_columns)}")
        else:
            print("attention: ordre des colonnes manquant")
except Exception as e:
    print(f"erreur chargement: {e}")
    model = None
    le = None
    feature_columns = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "label_encoder_loaded": le is not None,
        "feature_columns_loaded": feature_columns is not None,
        "num_features": len(feature_columns) if feature_columns else 0,
        "classes": list(le.classes_) if le is not None else []
    })

@app.route('/api/predict', methods=['POST'])
def predict_audio():
    if model is None:
        return jsonify({"error": "Modèle non chargé sur le serveur. Entraînez le modèle d'abord."}), 500

    if 'audio' not in request.files:
        return jsonify({"error": "Aucun fichier audio n'a été fourni"}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({"error": "Fichier audio vide"}), 400

    try:
        file_id = str(uuid.uuid4())
        temp_wav_path = os.path.join(TEMP_DIR, f"{file_id}.wav")
        
        audio_file.save(temp_wav_path)

        row_dict = {"path": temp_wav_path, "gender": "unknown"}
        raw_features = extract_features_spark(row_dict)

        if raw_features is None:
            return jsonify({"error": "Impossible d'extraire les caractéristiques audio. Audio trop silencieux ou corrompu."}), 400

        df_features = pd.DataFrame([raw_features])
        cols_to_drop = [c for c in ["gender", "path"] if c in df_features.columns]
        if cols_to_drop:
            df_features = df_features.drop(columns=cols_to_drop)

        if feature_columns:
            missing_cols = set(feature_columns) - set(df_features.columns)
            if missing_cols:
                print(f"colonnes manquantes: {missing_cols}")
                return jsonify({"error": f"colonnes manquantes: {missing_cols}"}), 400

            df_features = df_features[feature_columns]
        else:
            print("Attention: Ordre des colonnes non disponible, prédiction peut être incorrecte")

        prediction_idx = model.predict(df_features)[0]
        prob = model.predict_proba(df_features)[0]
        
        label = le.inverse_transform([prediction_idx])[0]
        confidence = prob[prediction_idx] * 100
        
        details = {class_name: float(prob[i]*100) for i, class_name in enumerate(le.classes_)}

        try:
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
        except Exception as e:
            print(f"erreur suppression temp: {e}")

        return jsonify({
            "success": True,
            "prediction": label,
            "confidence": round(confidence, 2),
            "details": details
        })

    except Exception as e:
        print(f"erreur prediction: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
