import numpy as np
import pickle as pk

from src.TraningPyFile.split_glass_csv import split_glass


class MLR:
    def __init__(self, alpha, iteration):
        self.alpha = alpha
        self.iteration = iteration
        self.weights = 0
        self.bias = 0

    def intializeWeights(self, x):
        self.weights = np.random.random((9, 7))
        self.bias = np.random.random((1, 7))

    def softmax(self, ypred):
        n = np.exp(ypred - np.max(ypred))
        d = np.sum(n, axis=0)
        sf = n / d
        return sf

    def yprediction(self, x):
        vec1 = np.dot(x, self.weights)
        ypred = np.add(vec1, self.bias)
        # ypred = np.add(vec2, -vec2.max(axis=0))
        # print(ypred.shape)
        yhat = self.softmax(ypred)
        return yhat

    def fit(self, x, y):
        self.intializeWeights(x)
        for i in range(self.iteration):
            # print(i)
            ypred = self.yprediction(x)
            p = np.argmax(ypred, axis=1)
            diff = y - p
            print("acc " + str(self.accuracy(x, y)))
            change_in_weights = np.mean(np.dot(diff, x), axis=0, keepdims=True)
            change_in_bias = np.mean(diff)
            self.weights = self.weights - (self.alpha * change_in_weights)
            self.bias = self.bias - (self.alpha * change_in_bias)

    def accuracy(self, x, y):
        ypred = self.predict(x)
        p = np.argmax(ypred, axis=1)
        accuracy = np.sum(p == y) * 100 / y.shape[0]
        return accuracy

    def predict(self, x):
        ypred = self.yprediction(x)
        return ypred


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = split_glass().split_glass_data()
    log = MLR(0.01, 15)
    # y_train = np.array(y_train)
    # y_test = np.array(y_test)
    log.fit(x_train, y_train)
    accuracy = log.accuracy(x_test, y_test)

    with open("/AI/venv/src/TrainedModels/MultipleLogisticRegression.pkl", 'wb') as f:
        pk.dump(log, f)
    with open("/AI/venv/src/TrainedModels/MultipleLogisticRegression.pkl", 'rb') as f:
        l = pk.load(f)
        accuracy = l.accuracy(x_test, y_test)
        print(accuracy)
