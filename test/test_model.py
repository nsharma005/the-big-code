from model_training.xg_boost import load_data, split_data, train_model
from sklearn.metrics import roc_auc_score, f1_score


def test_model_performance():
    df = load_data()

    X_train, X_test, y_train, y_test = split_data(df)

    model = train_model(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    assert auc > 0.7
    assert f1 > 0.6