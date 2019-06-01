import numpy as np

from src.TraningPyFile.Cifar_dataset import Cifar_dataset


class NearestNeighbor:
    def __init__(self):
        pass

    def train(self, x, y):
        self.x_train = x
        self.y_train = y

    def predict(self, x):
        total = x.shape[0]
        total=100
        y_prediction = np.zeros(total, dtype=self.y_train.dtype)
        for i in range(total):
            a = np.square(self.x_train - x[i, :])
            y = np.argmin(np.sqrt(np.sum(a, axis=0)))
            y_prediction[i] = self.y_train[y]
            print(y_prediction)
        return y_prediction

    def accuracy(self, y_pred, y_test):

        correct = np.sum(y_pred == y_test)
        print(correct)
        model_accuracy = float(correct) / y_test.shape[0]
        return model_accuracy


cifar = Cifar_dataset()
# x_train, y_train, x_test, y_test = cifar.cifar10_data_unpickling("cifar-10-batches-py")
x_train, y_train, x_test, y_test = cifar.cifar10_data_unpickling("/AI//venv//src//cifar-10-batches-py")
# print(len(data))

NN = NearestNeighbor()
NN.train(x_train, y_train)
y_pred = NN.predict(x_test)
accuracy = NN.accuracy(y_pred, y_test)
print(accuracy)
