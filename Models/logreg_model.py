from sklearn.linear_model import LogisticRegression

def get_model(C=1.0, max_iter=1000, random_state=2025):
    model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state, n_jobs=-1)
    return model
