import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump
from Models.rf_model import get_model as get_rf
from Models.svm_model import get_model as get_svm
from Models.knn_model import get_model as get_knn
from Models.logreg_model import get_model as get_logreg
import logging

# 超参
LOG_DIR = "./log/train_ml"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "train_ml.txt")


# 日志配置
logging.basicConfig(
    filename=LOG_FILE,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)



df = pd.read_csv("./data_process/preprocessed.csv")
feature_cols = [
    'total_packets', 'total_bytes', 'mean_size', 'std_size',
    'size_bin_0', 'size_bin_1', 'size_bin_2', 'size_bin_3', 'size_bin_ratio'
]
X = df[feature_cols]
y = df['page_label']


# 数据划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=2025, stratify=y
)


models = {
    "RandomForest": ("rf", get_rf()),
    "SVM": ("svm", get_svm()),
    "KNN": ("knn", get_knn()),
    "LogisticRegression": ("logreg", get_logreg())
}

# 训练 & 评估
results = {}
for name, (abbr, model) in models.items():
    logging.info(f"Training model: {name}")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    # 日志
    logging.info(f"{name} Test Accuracy: {acc:.4f}")
    logging.info(f"{name} Classification Report:\n{classification_report(y_test, y_pred, digits=4)}")
    print(f"\n{name} Test Accuracy: {acc:.4f}")

    # 保存模型
    model_path = os.path.join(LOG_DIR, f"ml_{abbr}.pkl")
    dump(model, model_path)
    logging.info(f"{name} model saved to: {model_path}")


# 输出最终结果
logging.info("Final Results Summary:")
for name, acc in results.items():
    logging.info(f"{name}: Test Accuracy = {acc:.4f}")

print("\nAll models and logs saved to ./log/train_ml/")
