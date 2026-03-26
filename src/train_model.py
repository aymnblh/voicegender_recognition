"""
train_model.py
charge les features parquet, entraine les classifieurs
evalue les performances et sauvegarde le meilleur modele
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
    ConfusionMatrixDisplay, roc_auc_score, roc_curve
)
from sklearn.pipeline import Pipeline

FEATURES_PARQUET = "output/features.parquet"
MODEL_OUTPUT  = "models/best_model.pkl"
TEST_SIZE     = 0.2
RANDOM_STATE  = 42


def load_data():
    print(f"chargement: {FEATURES_PARQUET}")

    if not os.path.exists(FEATURES_PARQUET):
        print(f"erreur: fichier non trouve")
        return None, None, None

    df = pd.read_parquet(FEATURES_PARQUET)

    df = df[df["gender"].isin(["male", "female"])].dropna()

    X = df.drop(columns=["gender", "path"])
    y = df["gender"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    print(f"distribution:\n{pd.Series(y).value_counts().to_string()}\n")
    return X, y_enc, le


def build_pipelines():
    """modeles avec scaler + classifieur"""
    return {
        "logistic regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
        ]),
        "svm rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE))
        ]),
        "random forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE))
        ]),
        "gradient boosting": Pipeline([
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

    print("-" * 60)
    print(f"{'modele':<25} | {'cv acc':>8} | {'test acc':>9} | {'auc':>7}")
    print("-" * 60)

    best_name, best_pipe, best_auc = None, None, -1.0
    
    plt.figure(figsize=(10, 8))

    for name, pipe in pipelines.items():
        cv_scores = cross_val_score(pipe, X_trainval, y_trainval, cv=5,
                                    scoring="accuracy", n_jobs=-1)
        pipe.fit(X_trainval, y_trainval)
        
        y_pred  = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        
        acc  = np.mean(y_pred == y_test)
        auc  = roc_auc_score(y_test, y_proba)
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (auc = {auc:.3f})")

        print(f"{name:<25} | {cv_scores.mean():.4f}   | {acc:.4f}    | {auc:.4f}")
        results[name] = {"pipeline": pipe, "acc": acc, "auc": auc,
                         "y_pred": y_pred, "y_proba": y_proba}

        if auc > best_auc:
            best_auc, best_name, best_pipe = auc, name, pipe

    print("-" * 60)
    print(f"meilleur modele: {best_name} (auc={best_auc:.4f})\n")

    os.makedirs("models", exist_ok=True)
    os.makedirs("models/visualizations", exist_ok=True)

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label="chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('faux positifs')
    plt.ylabel('vrais positifs')
    plt.title('courbes roc')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("models/visualizations/roc_curves.png", dpi=150)
    plt.close()
    print("roc curves: models/visualizations/roc_curves.png")

    print(f"rapport: {best_name}")
    print(classification_report(y_test, results[best_name]["y_pred"],
                                 target_names=le.classes_))

    cm = confusion_matrix(y_test, results[best_name]["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap="Blues")
    plt.title(f"confusion matrix - {best_name}")
    plt.tight_layout()
    plt.savefig("models/visualizations/confusion_matrix.png", dpi=150)
    print("confusion matrix: models/visualizations/confusion_matrix.png")

    if "random forest" in results:
        rf_pipe = results["random forest"]["pipeline"]
        rf_clf  = rf_pipe.named_steps["clf"]
        importances = pd.Series(rf_clf.feature_importances_, index=X.columns)
        top20 = importances.nlargest(20)
        
        plt.figure(figsize=(10, 8))
        top20.sort_values().plot(kind="barh", color="#2E86C1")
        plt.title("top 20 features (random forest)", fontsize=14)
        plt.xlabel("importance")
        plt.ylabel("features")
        plt.tight_layout()
        plt.savefig("models/visualizations/feature_importance.png", dpi=150)
        plt.close()
        print("feature importance: models/visualizations/feature_importance.png\n")

    return best_name, best_pipe, X


def main():
    print(f"chargement...")

    if not os.path.exists(FEATURES_PARQUET):
        print(f"erreur: {FEATURES_PARQUET} non trouve")
        return
        
    X, y, le = load_data()
    print(f"dataset: {X.shape[0]} exemples x {X.shape[1]} features\n")

    best_name, best_pipe, X_train = train_and_evaluate(X, y, le)

    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "model": best_pipe,
        "label_encoder": le,
        "feature_columns": list(X_train.columns)
    }, MODEL_OUTPUT)
    print(f"modele sauvegarde: {MODEL_OUTPUT}")
    print(f"colonnes: {len(X_train.columns)} features")


if __name__ == "__main__":
    main()
