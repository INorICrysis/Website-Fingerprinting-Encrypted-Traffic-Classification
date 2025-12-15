import torch
from torch.utils.data import Dataset, DataLoader

class Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloader(X_train, y_train, X_test, y_test, batch_size=128):
    train_loader = DataLoader(Dataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(Dataset(X_test, y_test),
                             batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
