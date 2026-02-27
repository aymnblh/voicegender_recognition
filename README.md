# Voice Gender Recognition

Ce projet utilise l'apprentissage automatique (Machine Learning) pour identifier le genre d'une voix à partir de fichiers audio WAV.

## Structure du projet

- `src/main.py` : Script de base pour l'exploration initiale des fichiers.
- `src/feature_extraction.py` : Extrait des caractéristiques riches (MFCC, Spectral, Chroma, Pitch) en utilisant `librosa`.
- `src/train_model.py` : Entraîne plusieurs modèles (Random Forest, SVM, etc.) et sauvegarde le meilleur.
- `models/` : Contient le modèle entraîné et les rapports visuels.

## Dépendances

- librosa
- pandas
- scikit-learn
- joblib
- tqdm
- matplotlib

## Utilisation

1. **Extraction** : `python src/feature_extraction.py`
2. **Entraînement** : `python src/train_model.py`
