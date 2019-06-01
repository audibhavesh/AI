import pickle as pk

from src.TraningPyFile.Multiple_Logistic_Regression import MLR
from src.TraningPyFile.gradientdecent_LR import LinearRegression
from src.TraningPyFile.split_breast_cancer_csv import Split
from src.TraningPyFile.split_glass_csv import split_glass
# with open("/AI/venv/src/TrainedModels/LRmodel.pkl", 'rb') as f:
#     d = pk.load(f)
#     print(d.predict(67))
# x_train, y_train, x_test, y_test = Split().split_xy()
# with open("/AI/venv/src/TrainedModels/LogisticRegression.pkl", 'rb') as f:
#     l = pk.load(f)
#     loss = l.loss(x_test, y_test)
#     accuracy = l.accuracy(x_test, y_test)
#     print(loss, accuracy * 100)

with open("/AI/venv/src/TrainedModels/MultipleLogisticRegression.pkl", 'rb') as f:
    l=pk.load(f)
    x_train, y_train, x_test, y_test = split_glass().split_glass_data()
    p=l.predict(x_train[75])
    print(p)
    print(y_train[75])

