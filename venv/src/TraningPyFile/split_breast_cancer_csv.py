import pandas as pd
import numpy as np


class Split:
    def __init__(self):
        pass

    def split_xy(self, split=0.2):
        df = pd.read_csv('/AI/venv/src/datasets/breast-cancer.csv')
        df.dropna(axis='columns',inplace=True)
        x = df[df.columns[2:]].values
        y = df['diagnosis'].values
        y= (y == 'M').astype('float')
        # print(y)
        y = np.expand_dims(y, -1)
        split=400
        x_train = x[:split]
        x_test = x[split:]
        y_train = y[:split]
        y_test = y[split:]
        return x_train, y_train, x_test, y_test


