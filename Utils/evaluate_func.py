import os
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return acc, all_preds, all_labels