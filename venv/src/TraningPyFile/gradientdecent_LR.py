import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as an
import pickle as pk


class LinearRegression:
    def __init__(self, alpha, iteration):
        # learning rate
        self.alpha = alpha
        # no of itration on data
        self.iteration = iteration
        self.slope = 0
        # biases

        self.intercept = 0

    def predict(self, x):
        y_perd = np.dot(x, self.slope) + self.intercept
        return y_perd

    def calculate_gradientdecent(self, m, b, x, y):
        change_in_slope = 0.0
        change_in_intercept = 0.0
        N = float(len(x))
        for i in range(len(x)):
            y_pred = (m * x[i]) + b
            diff = y_pred-y[i]
            change_in_slope += x[i] * diff
            change_in_intercept += diff
        new_slope = m - (self.alpha * change_in_slope*(1/N))
        new_intercept = b - (self.alpha * change_in_intercept*(1/N))
        return new_slope, new_intercept
    def animate(self,x,y):
        pass

    def fit(self, x_train, y_train):
        for i in range(self.iteration):
            print(i)
            self.slope,self.intercept=self.calculate_gradientdecent(self.slope,self.intercept,x_train,y_train)
        return self.slope, self.intercept

    def error(self, y, y_pred):
        diff = y - y_pred
        MSE = np.mean(diff ** 2)
        return MSE

if __name__=="__main__":

    lr = LinearRegression(0.0001, 1300)
    df = pd.read_csv('datasets/weight-height.csv')
    x_data = np.array(df["Height"])
    y_data = np.array(df['Weight'])
    x_train=x_data[0:7000]
    y_train=y_data[0:7000]
    x_test=x_data[7000:]
    y_test=y_data[7000:]
    m, b = lr.fit(x_train, y_train)
    # print(m, b)
    y=lr.predict(x_test)
    print(y)
    with open("LRmodel.pkl",'wb') as f:
        pk.dump(lr, f)
    print(lr.error(y_test,y))
    with open("LRmodel.pkl", 'rb') as f:
        d = pk.load(f)
        print(d.predict(65))
