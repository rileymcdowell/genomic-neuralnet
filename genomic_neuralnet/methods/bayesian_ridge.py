from sklearn.linear_model import BayesianRidge

def get_brr_prediction(train_data, train_truth, test_data, test_truth):
    clf = BayesianRidge()
    clf.fit(train_data, train_truth)
    predicted = clf.predict(test_data)
    return predicted.ravel()
