import numpy as np
from src.TraningPyFile.Cifar_dataset import Cifar_dataset


class kNearestNeighbor:
    def __init__(self):
        pass

    def train(self, x, y):
        self.x_train = x
        self.y_train = y

    def predict(self, x, y, k=1):
        total = x.shape[0]
        total = 100
        # y_prediction = np.zeros(total, dtype=self.y_train.dtype)
        y_prediction = []
        for i in range(total):
            a = np.square(x[i, :] - self.x_train)
            distance = np.sqrt(np.sum(a, axis=0))
            neighbors = np.sort(distance)[:k]
            knearest = np.argmin(neighbors)
            print(neighbors.shape)
            y_prediction.append(self.y_train[knearest])
            # print(y_prediction)
            print(i)
            # if i in list(range(1000,10000,1000)):
            #     self.accuracy(y_pred,y[0:i])
            self.accuracy(y_prediction, y[0:i])

        return y_prediction

    def accuracy(self, y_pred, y_test):
        correct = np.sum(y_pred == y_test)
        print("correct " + str(correct))
        # model_accuracy = float(correct) / y_test.shape[0]
        # print("accuracy "+ str(model_accuracy))
        # return model_accuracy


cifar = Cifar_dataset()
x_train, y_train, x_test, y_test = cifar.cifar10_data_unpickling("/AI//venv//src//cifar-10-batches-py")
NN = kNearestNeighbor()
NN.train(x_train, y_train)
y_pred = NN.predict(x_test, y_test, 3)
accuracy = NN.accuracy(y_pred, y_test)
print(accuracy)
