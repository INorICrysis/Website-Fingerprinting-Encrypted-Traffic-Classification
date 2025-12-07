import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from Utils.rf_model import get_model as get_rf
from Utils.svm_model import get_model as get_svm
from Utils.knn_model import get_model as get_knn
from Utils.logreg_model import get_model as get_logreg

# 读数据
df = pd.read_csv("./data_process/preprocessed.csv")

# Feature
feature_cols = [
    'total_packets', 'total_bytes', 'mean_size', 'std_size',
    'size_bin_0', 'size_bin_1', 'size_bin_2', 'size_bin_3', 'size_bin_ratio'
]
X = df[feature_cols]

# Target
y = df['page_label']


# 数据划分
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=2025, stratify=y
)


# 待使用的模型字典
models = {
    "RandomForest": get_rf(),
    "SVM": get_svm(),
    "KNN": get_knn(),
    "LogisticRegression": get_logreg()
}

# train
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"\n{name} Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, digits=4))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
