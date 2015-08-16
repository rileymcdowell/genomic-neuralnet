from sklearn.linear_model import Ridge
import numpy as np

def get_rr_prediction(train_data, train_truth, test_data, test_truth, alpha=1.0):
    clf = Ridge(alpha=alpha, solver='lsqr') # Modern scipy least-squares solver.
    clf.fit(train_data, train_truth)
    predicted = clf.predict(test_data)
    return predicted.ravel()
