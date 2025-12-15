from sklearn.neighbors import KNeighborsClassifier

def get_model(n_neighbors=5):
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    return model