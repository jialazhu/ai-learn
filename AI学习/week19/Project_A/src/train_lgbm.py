import json
import pathlib
import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import roc_auc_score
import onnxmltools
from onnxmltools.convert import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType


DATA_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "processed"
MODEL_DIR = pathlib.Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    train_df = pd.read_parquet(DATA_DIR / "train.parquet")
    test_df = pd.read_parquet(DATA_DIR / "test.parquet")
    # 特征列顺序
    with open(DATA_DIR / "columns.json", "r", encoding="utf-8") as f:
        feature_names = json.load(f)

    y_train = train_df["Class"].values
    X_train = train_df[feature_names].values
    y_test = test_df["Class"].values
    X_test = test_df[feature_names].values
    return X_train, y_train, X_test, y_test, feature_names


def train_lgbm(X_train, y_train, X_val, y_val):
    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": -1,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "is_unbalance": True,
    }
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    model = lgb.train(
        params,
        lgb_train,
        num_boost_round=500,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=50),
        ],
    )
    return model


def main():
    X_train, y_train, X_test, y_test, feature_names = load_data()
    model = train_lgbm(X_train, y_train, X_test, y_test)
    preds = model.predict(X_test, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_test, preds)
    print(f"Test AUC: {auc:.4f}")

    model_path = MODEL_DIR / "lgbm_model.txt"
    model.save_model(model_path)
    joblib.dump(model, MODEL_DIR / "lgbm_model.pkl")
    print("Saved model to:", model_path)

    # 导出 ONNX 便于 CPU 推理/量化后推理
    try:
        # 使用 onnxmltools 自带的 FloatTensorType
        initial_type = [("input", FloatTensorType([None, len(feature_names)]))]
        onnx_model = convert_lightgbm(model, initial_types=initial_type, target_opset=15)
        onnx_path = MODEL_DIR / "lgbm_model.onnx"
        onnxmltools.utils.save_model(onnx_model, str(onnx_path))
        print("Saved ONNX model to:", onnx_path)
    except Exception as e:
        print("ONNX export failed:", e)


if __name__ == "__main__":
    main()

