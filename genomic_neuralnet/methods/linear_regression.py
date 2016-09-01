from sklearn.linear_model import LinearRegression

def get_lr_prediction(train_data, train_truth, test_data, test_truth, iter_id=0):
    clf = LinearRegression() 
    clf.fit(train_data, train_truth)
    predicted = clf.predict(test_data)
    return predicted.ravel()
