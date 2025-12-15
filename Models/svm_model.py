from sklearn.svm import SVC

def get_model(kernel='rbf', C=1.0, random_state=2025):
    model = SVC(kernel=kernel, C=C, random_state=random_state)
    return model
