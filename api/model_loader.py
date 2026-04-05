import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "xgb_model.pkl"

model = None

def load_model():
    global model
    if model is None:
        model = joblib.load(MODEL_PATH)
    return model

def load_shap_explainer():
    import joblib
    from pathlib import Path

    MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
    return joblib.load(MODEL_DIR / "shap_explainer.pkl")