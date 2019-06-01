import numpy as np
import pickle as pk
from src.TraningPyFile.split_breast_cancer_csv import Split


class LogisticRegression:
    def __init__(self, alpha, iteration):
        self.alpha = alpha
        self.iteration = iteration
        self.weights = 0
        self.bias = 0

    def intializeWeights(self, x):
        lis = [np.random.rand(1, 1) for i in range(x.shape[1])]
        self.weights = np.array(lis).reshape((30, 1))
        self.bias = 0

    def activationfunction(self, ypred):
        # Sigmoid Function
        # s = 1 / (1 + np.exp(-ypred))
        s = .5 * (1 + np.tanh(.5 * ypred))
        return s

    def yprediction(self, x):
        # using y=wx+b
        ypred = np.dot(x, self.weights) + self.bias
        return ypred

    def fit(self, x_train, y_train):
        self.intializeWeights(x_train)
        x_train = self.normalize(x_train)
        for i in range(self.iteration):
            ypred = self.activationfunction(self.yprediction(x_train))
            diff =ypred-y_train
            change_in_weights = np.mean(diff * x_train, axis=0,keepdims=True).T
            change_in_bias = np.mean(diff)
            self.weights = self.weights - (self.alpha * change_in_weights)
            self.bias = self.bias - (self.alpha * change_in_bias)
        return self

    def normalize(self, x):
        self.xmean = x.mean(axis=0).T
        self.standardeviation = x.std(axis=0).T
        x = (x - self.xmean) / self.standardeviation
        return x

    def loss(self, x, y):
        # cross-Entropy
        ypred = self.activationfunction(self.yprediction(x))
        y_benign = y * np.log(ypred + 1e-9)
        y_malignant = (1 - y) * np.log((1 - ypred) + 1e-9)
        h_mb = -np.mean(y_benign + y_malignant)
        return h_mb

    def accuracy(self, x, y):
        # using mean
        ypred = self.predict(x)
        accuracy = np.mean(ypred == y)
        return accuracy

    def predict(self, x):
        x = self.normalize(x)
        ypred = self.activationfunction(self.yprediction(x))
        return (ypred >= 0.5).astype('int')


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = Split().split_xy()

    log = LogisticRegression(0.01, 500)
    log.fit(x_train, y_train)
    loss = log.loss(x_test, y_test)
    accuracy = log.accuracy(x_test, y_test)
    print(loss, accuracy * 100)
    # with open("/AI/venv/src/TrainedModels/LogisticRegression.pkl", 'wb') as f:
    #     pk.dump(log, f)
    with open("/AI/venv/src/TrainedModels/LogisticRegression.pkl", 'rb') as f:
        l = pk.load(f)
        loss = l.loss(x_test, y_test)
        accuracy = l.accuracy(x_test, y_test)
        print(loss, accuracy * 100)
