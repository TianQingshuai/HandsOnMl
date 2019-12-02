import tensorflow as tf
import numpy as np
from tensorflow import keras
import os


class Regressor(keras.layers.Layer):
    def __init__(self, dim_in, dim_out):
        super(Regressor, self).__init__()

        self.w = self.add_variable('w', [dim_in, dim_out])
        self.b = self.add_variable('b', [dim_out])
        print(self.w.shape, self.b.shape)
        print(type(self.w), tf.is_tensor(self.w), self.w.name)

    def call(self, x):
        y_pred = tf.matmul(x, self.w) + self.b
        return y_pred


def main():
    tf.random.set_seed(1)
    np.random.seed(1)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    assert tf.__version__.startswith('2')

    (x_train, y_train), (x_val, y_val) = keras.datasets.boston_housing.load_data()
    x_train, x_val = x_train.astype(np.float32), x_val.astype(np.float32)
    print(x_train.shape, x_val.shape, y_train.shape, y_val.shape)
    dim_in = x_train.shape[1]
    print("dim_in:", dim_in)

    db_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(64)
    db_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(102)


    model = Regressor(dim_in, 1)
    criteon = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate = 1e-2)

    for epoch in range(200):
        for step, (x,y) in enumerate(db_train):
            with tf.GradientTape() as tape:
                logits = model(x)
                logits = tf.squeeze(logits, axis=1)
                loss = criteon(y, logits)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print(epoch, 'loss:', loss.numpy())

        if epoch % 10 == 0:
            logits = model(x)
            logits = tf.squeeze(logits, axis=1)
            loss = criteon(y, logits)
            print(epoch, 'val loss:', loss.numpy())

if __name__ == '__main__':
    main()







