import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from Models.mlp import MLP
from Utils.dataloader import get_dataloader
from Utils.train_func import train
from Utils.evaluate_func import evaluate





# 超参
DATA_CSV = "./data_process/preprocessed.csv"
TEST_SIZE = 0.25
RANDOM_STATE = 2025
BATCH_SIZE = 128
EPOCHS = 500
LR = 1e-4
MLP_HIDDEN1 = 128
MLP_HIDDEN2 = 64
MLP_DROPOUT1 = 0.3
MLP_DROPOUT2 = 0.2
RF_N_ESTIMATORS = 200
SEED = 2025

# 种子设置
torch.manual_seed(SEED)
np.random.seed(SEED)

# 日志设置
LOG_DIR = "./log/train_mlp"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "train_mlp.txt")
MODEL_FILE = os.path.join(LOG_DIR, "mlp_best_model.pth")





# 加载数据
df = pd.read_csv(DATA_CSV)
feature_cols = [
    'total_packets','total_bytes','mean_size','std_size',
    'size_bin_0','size_bin_1','size_bin_2','size_bin_3','size_bin_ratio'
]
X = df[feature_cols].values.astype(np.float32)
y = df['page_label'].values.astype(np.int64)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)





# DataLoader & 模型构建
train_loader, test_loader = get_dataloader(X_train, y_train, X_test, y_test, batch_size=BATCH_SIZE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mlp = MLP(
    in_dim=X_train.shape[1],
    num_classes=len(np.unique(y)),
    hidden1=MLP_HIDDEN1,
    hidden2=MLP_HIDDEN2,
    dropout1=MLP_DROPOUT1,
    dropout2=MLP_DROPOUT2
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mlp.parameters(), lr=LR)





# train
best_acc = 0.0
best_model_state = None
with open(LOG_FILE, "w") as log:
    log.write(f"Using device: {device}\n")
    log.write(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}\n\n")

    for epoch in range(EPOCHS):
        loss = train(mlp, train_loader, criterion, optimizer)
        acc, _, _ = evaluate(mlp, test_loader)

        if acc > best_acc:
            best_acc = acc
            best_model_state = mlp.state_dict()
            torch.save(best_model_state, MODEL_FILE)

        log_line = f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss:.4f}, Test Acc: {acc:.4f}\n"
        print(log_line.strip())
        log.write(log_line)





# 结果打印
mlp.load_state_dict(torch.load(MODEL_FILE))
acc, preds, labels = evaluate(mlp, test_loader)
print(f"\nFinal Test Accuracy (Best Model): {acc:.4f}")
final_report = classification_report(labels, preds, digits=4)
print(final_report)

with open(LOG_FILE, "a") as log:
    log.write("\nFinal Test Accuracy (Best Model): {:.4f}\n".format(acc))
    log.write(final_report + "\n")
