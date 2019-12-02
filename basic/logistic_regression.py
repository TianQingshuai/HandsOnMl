from __future__ import absolute_import, division, print_function, unicode_literals
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
import pandas 
import functools

class LogisticRegressor(keras.layers.Layer):
    def __init__(self, dim_in, dim_out):
        super(LogisticRegressor, self).__init__()

        self.w = self.add_variable('w', [dim_in, dim_out])
        self.b = self.add_variable('b', [dim_out])

    def call(self, x):
        z = tf.matmul(x, self.w) + self.b

        y_pred = 1 / (1 + tf.math.exp(-z))
        return y_pred

def get_dataset(epochs):
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../demo/Iris.csv')
    print(file_path)
    header = ['id', 'width', 'height', 'label']
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size = 50,
        column_names = header,
        select_columns = ['width', 'height', 'label'],
        label_name = 'label',
        header = False,
        num_epochs = epochs
    )
    return dataset


def main():
    # IRIS_DATA_URL = "https://media.geeksforgeeks.org/wp-content/uploads/dataset.csv"
    epochs = 100
    ds = get_dataset(epochs)
    model = LogisticRegressor(2, 1)





if __name__  == '__main__':
    main()