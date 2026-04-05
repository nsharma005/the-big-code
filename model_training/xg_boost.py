import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from config.features import FEATURE_COLUMNS
import shap
import matplotlib.pyplot as plt
import xgboost as xgb
import joblib
import numpy as np

# PATHS
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
GENERATED_DIR = BASE_DIR / "data" / "generated"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

# LOAD DATA
def load_data():
    file_path = GENERATED_DIR / "user_features.csv"
    print("Reading from:", file_path)
    return pd.read_csv(file_path)

# USER-LEVEL SPLIT
def split_data(df):
    user_ids = df["user_id"].unique()

    train_users, test_users = train_test_split(
        user_ids, test_size=0.2, random_state=42
    )

    train_df = df[df["user_id"].isin(train_users)]
    test_df = df[df["user_id"].isin(test_users)]

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["is_bot"]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["is_bot"]

    return X_train, X_test, y_train, y_test

# MODEL
def train_model(X_train, y_train):
    base_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="auc"
    )

    model = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
    model.fit(X_train, y_train)
    return model

# EVALUATION
def evaluate(model, X_test, y_test):

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


    # CORE METRICS

    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\nAdvanced Metrics:")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall (Bot Detection Power): {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")


    # BUSINESS METRICS


    # % of fake engagement detected
    fake_detected = (y_pred == 1).sum() / len(y_pred)

    # average predicted authenticity
    avg_auth = (1 - y_prob).mean()

    # Precision@K (top suspicious users)
    k = int(0.1 * len(y_prob))  # top 10%
    top_k_idx = y_prob.argsort()[-k:]

    precision_at_k = y_test.iloc[top_k_idx].mean()

    print("\n💼 Business Metrics:")
    print(f"Fake Engagement Detected: {fake_detected*100:.2f}%")
    print(f"Avg Authenticity Score: {avg_auth*100:.2f}%")
    print(f"Precision@Top10%: {precision_at_k:.4f}")

# FEATURE IMPORTANCE
def feature_importance(model, X_train):
    if hasattr(model, "calibrated_classifiers_"):
        base_model = model.calibrated_classifiers_[0].estimator
    else:
        base_model = model

    importances = base_model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": X_train.columns,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    print("\nFeature Importance:")
    print(importance_df)

# ERROR ANALYSIS
def error_analysis(X_test, y_test, model):

    df = X_test.copy()
    df["true"] = y_test.values
    df["pred"] = model.predict(X_test)

    false_positives = df[(df["true"] == 0) & (df["pred"] == 1)]
    false_negatives = df[(df["true"] == 1) & (df["pred"] == 0)]

    print("\nError Analysis:")
    print("False Positives (real users flagged):", len(false_positives))
    print("False Negatives (bots missed):", len(false_negatives))

# SHAP ANALYSIS (FIXED)
def shap_analysis(model, X_train):
    print("\n🔍 Running SHAP Analysis...")

    # Extract base model
    if hasattr(model, "calibrated_classifiers_"):
        base_model = model.calibrated_classifiers_[0].estimator
    else:
        base_model = model

    # Sample data (performance safe)
    X_sample = X_train.sample(min(1000, len(X_train)), random_state=42)

    # FIX: removed model_output="probability"
    explainer = shap.TreeExplainer(base_model)

    shap_values = explainer.shap_values(X_sample)

    # Save SHAP plot
    try:
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.savefig(GENERATED_DIR / "shap_summary.png")
        plt.clf()
        print("SHAP plot saved")
    except:
        print("Plot skipped")

    return explainer

# RUN
def run():
    df = load_data()

    X_train, X_test, y_train, y_test = split_data(df)

    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    model = train_model(X_train, y_train)

    evaluate(model, X_test, y_test)
    feature_importance(model, X_train)
    error_analysis(X_test, y_test, model)

    explainer = shap_analysis(model, X_train)

    # Save artifacts
    joblib.dump(model, MODEL_DIR / "xgb_model.pkl")
    joblib.dump(explainer, MODEL_DIR / "shap_explainer.pkl")

    print(f"\nModel saved to: {MODEL_DIR / 'xgb_model.pkl'}")
    print(f"Explainer saved to: {MODEL_DIR / 'shap_explainer.pkl'}")


    print("\nFEATURE ORDER:")
    print(X_train.columns.tolist())

if __name__ == "__main__":
    run()