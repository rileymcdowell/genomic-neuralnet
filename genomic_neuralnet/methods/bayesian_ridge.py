from sklearn.linear_model import BayesianRidge

def get_brr_prediction(train_data, train_truth, test_data, test_truth, alpha_1, alpha_2, lambda_1, lambda_2):
    clf = BayesianRidge(alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2)
    clf.fit(train_data, train_truth)
    predicted = clf.predict(test_data)
    return predicted.ravel()
