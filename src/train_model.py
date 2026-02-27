"""
train_model.py
--------------
Charge features.csv, entraîne plusieurs classifieurs,
évalue leurs performances et sauvegarde le meilleur modèle.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score
)
from sklearn.pipeline import Pipeline

FEATURES_CSV  = "features.csv"
MODEL_OUTPUT  = "models/best_model.pkl"
TEST_SIZE     = 0.2
VAL_SIZE      = 0.1   
RANDOM_STATE  = 42


def load_data():
    df = pd.read_csv(FEATURES_CSV)
    df = df[df["gender"].isin(["male", "women"])].dropna()

    X = df.drop(columns=["gender", "path"])
    y = df["gender"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"Classes : {dict(zip(le.classes_, le.transform(le.classes_)))}")
    print(f"Distribution :\n{pd.Series(y).value_counts().to_string()}\n")
    return X, y_enc, le


def build_pipelines():
    """Retourne un dict de Pipelines (StandardScaler + classifieur)."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_STATE))
        ]),
    }


def train_and_evaluate(X, y, le):
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipelines = build_pipelines()
    results = {}

    print("=" * 60)
    print(f"{'Modèle':<25} | {'CV Acc':>8} | {'Test Acc':>9} | {'AUC':>7}")
    print("=" * 60)

    best_name, best_pipe, best_auc = None, None, -1.0

    for name, pipe in pipelines.items():
        # Cross-validation sur train+val
        cv_scores = cross_val_score(pipe, X_trainval, y_trainval, cv=5,
                                    scoring="accuracy", n_jobs=-1)
        # Entraînement final sur tout trainval
        pipe.fit(X_trainval, y_trainval)
        # Évaluation sur test
        y_pred  = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        acc  = np.mean(y_pred == y_test)
        auc  = roc_auc_score(y_test, y_proba)

        print(f"{name:<25} | {cv_scores.mean():.4f}   | {acc:.4f}    | {auc:.4f}")
        results[name] = {"pipeline": pipe, "acc": acc, "auc": auc,
                         "y_pred": y_pred, "y_proba": y_proba}

        if auc > best_auc:
            best_auc, best_name, best_pipe = auc, name, pipe

    print("=" * 60)
    print(f"\n🏆 Meilleur modèle : {best_name}  (AUC={best_auc:.4f})\n")

    # Rapport détaillé du meilleur modèle
    print(f"── Rapport de classification ({best_name}) ──")
    print(classification_report(y_test, results[best_name]["y_pred"],
                                 target_names=le.classes_))

    # Matrice de confusion
    os.makedirs("models", exist_ok=True)
    cm = confusion_matrix(y_test, results[best_name]["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap="Blues")
    plt.title(f"Matrice de confusion – {best_name}")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png", dpi=150)
    print("   → Matrice de confusion sauvegardée dans models/confusion_matrix.png")

    # Feature importance (Random Forest seulement)
    if "Random Forest" in results:
        rf_pipe = results["Random Forest"]["pipeline"]
        rf_clf  = rf_pipe.named_steps["clf"]
        importances = pd.Series(rf_clf.feature_importances_, index=X.columns)
        top20 = importances.nlargest(20)
        top20.sort_values().plot(kind="barh", figsize=(8, 6))
        plt.title("Top 20 features (Random Forest)")
        plt.tight_layout()
        plt.savefig("models/feature_importance.png", dpi=150)
        print("   → Importance des features sauvegardée dans models/feature_importance.png\n")

    return best_name, best_pipe


def main():
    print(f"Chargement de {FEATURES_CSV}...")
    X, y, le = load_data()
    print(f"Dataset : {X.shape[0]} exemples × {X.shape[1]} features\n")

    best_name, best_pipe = train_and_evaluate(X, y, le)

    # Sauvegarder le meilleur modèle + le label encoder
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": best_pipe, "label_encoder": le}, MODEL_OUTPUT)
    print(f"✅ Modèle sauvegardé dans {MODEL_OUTPUT}")


if __name__ == "__main__":
    main()
