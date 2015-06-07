from sklearn.linear_model import Ridge

def get_rr_prediction(train_data, train_truth, test_data, test_truth):
    clf = Ridge()
    clf.fit(train_data, train_truth)
    predicted = clf.predict(test_data)
    return predicted.ravel()
