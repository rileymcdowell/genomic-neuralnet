from sklearn.linear_model import ElasticNet

def get_en_prediction(train_data, train_truth, test_data, test_truth, alpha=1.0, l1_ratio=0.5):
    clf = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    clf.fit(train_data, train_truth)
    predicted = clf.predict(test_data)
    return predicted.ravel()
