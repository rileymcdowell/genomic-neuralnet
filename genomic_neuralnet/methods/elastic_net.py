from sklearn.linear_model import ElasticNet

def get_en_prediction(train_data, train_truth, test_data, test_truth):
    clf = ElasticNet()
    clf.fit(train_data, train_truth)
    predicted = clf.predict(test_data)
    return predicted.ravel()
