import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, alpha, itration):
        # learning rate
        self.alpha = alpha
        # no of itration on data
        self.iteration = itration

    def predict(self, x):
        x = self.normalize(x)
        y_perd = np.dot(x, self.slope) + self.intercept
        y_perd_unorm = (y_perd * self.y_std) + self.y_mean
        return y_perd_unorm

    def intialize_weights(self, x):
        # slope also known as weight in lr
        self.slope = np.random.rand(x.shape[0], )

        # biases
        self.intercept = np.zeros((1,))

    def fit(self, x_train, y_train):
        self.intialize_weights(x_train)
        self.x_mean = x_train.mean(axis=0).T
        self.x_std = x_train.std()
        self.y_mean = y_train.mean(axis=0).T
        self.y_std = y_train.std()

        x_train, y_train = self.normalize(x_train, y_train)
        for i in range(self.iteration):
            # print(i)
            # intial random prediction y=mx+c
            y_pred = np.dot(x_train, self.slope) + self.intercept
            diff = y_pred - y_train
            change_in_slope = np.mean(diff * x_train)
            change_in_intercept = np.mean(diff)
            self.slope = self.slope - (self.alpha * change_in_slope)
            self.intercept = self.intercept - (self.alpha * change_in_intercept)
        return self.slope

    def error(self, y, y_pred):
        diff = y - y_pred
        MSE = np.mean(diff ** 2)
        return MSE

    def accuracy(self, y, y_pred):
        correct = np.sum(y_pred == y)
        return float(correct) / y.shape[0]

    def normalize(self, x, y=None):

        xd = (x - self.x_mean) / self.x_std
        if y is None:
            return xd
        yd = (y - self.y_mean) / self.y_std
        # print(x,xd*self.x_std+self.x_mean)
        return xd, yd


lr = LinearRegression(1e-3, 1000)
df = pd.read_csv('datasets/weight-height.csv')
x_data = np.array(df["Height"])
y_data = np.array(df['Weight'])

x_train = x_data[0:5000]
# print(np.shape(x_train))

x_test = x_data[5000:]
y_train = y_data[0:5000]
y_test = y_data[5000:]
x_test = x_test

a = lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
print(y_pred)
print(lr.error(y_train, y_pred))
print(lr.error(y_test, y_pred))
print(lr.accuracy(y_test, y_pred))

# # x_train, y_train = lr.normalize(x_train, y_train)
plt.scatter(x_train,y_train)
plt.plot(x_test,y_pred)
# plt.xlim()
plt.show()