import pandas as pd
import numpy as np


class split_glass:
    def __init__(self):
        pass

    def split_glass_data(self, split=0.7):
        df = pd.read_csv('/AI/venv/src/datasets/glass.csv')
        x = df[df.columns[0:9]].values
        y = df['Type']
        split_ratio = round(len(y) * split)
        x_train = x[:split_ratio]
        y_train = y[:split_ratio]
        x_test = x[split_ratio:]
        y_test = y[split_ratio:]
        return x_train, y_train, x_test, y_test


split_glass().split_glass_data()
