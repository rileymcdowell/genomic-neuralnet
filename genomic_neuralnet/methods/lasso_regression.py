from sklearn.linear_model import Lasso

def get_lasso_prediction(train_data, train_truth, test_data, test_truth):
    clf = Lasso()
    clf.fit(train_data, train_truth)
    predicted = clf.predict(test_data)
    return predicted.ravel()
