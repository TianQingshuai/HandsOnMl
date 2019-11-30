# -*- encoding: utf-8
import os
import sys
import math
import pandas as pd
from collections import Counter
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pprint 
import matplotlib as mlp
import matplotlib.pyplot as plt

ratings = pd.read_csv('../demo/ml-20m/ratings.csv', 
    names=['userid', 'movieid', 'rating', 'timestamp'], skiprows=1, nrows=500000)
genres = pd.read_csv('../demo/ml-20m/movies.csv', 
    names=['movieid', 'movienm', 'genreid'], skiprows=1, nrows=1000)

dictionary = dict(zip(genres.movieid, genres.genreid))


# 
ratings['genres'] = ratings.movieid.map(dictionary)
ratings['genres'] = ratings.genres.map(lambda x : x.split('|') if not pd.isnull(x) else [x])


def unnest(df, col, reset_index=True):
    tlist = []
    for i, y in df[col].iteritems():
            for x in y:
                tlist.append([i, x])
    col_flat = pd.DataFrame(tlist, columns = ['I', col])
    col_flat = col_flat.set_index('I')
    df = df.drop(col, 1)
    df = df.merge(col_flat, left_index=True, right_index=True)
    if reset_index:
        df = df.reset_index(drop=True)
    return df
# 打平genres
ratings = unnest(ratings, 'genres')

ratings.set_index('userid', inplace=True)
ratings = ratings.drop('timestamp', 1)

# user only first
#x = Counter(ratings.index).most_common(10)
#top_k = dict(x).keys()
#ratings = ratings[ratings.index.isin(top_k)]
#ratings.head()

ratings['_userid'] = ratings.index
ratings['movieid'] = ratings.movieid.astype('category')
ratings['genres'] = ratings.genres.astype('category')
ratings['_userid'] = ratings._userid.astype('category')

trans_ratings = pd.get_dummies(ratings)

y = trans_ratings['rating'].values
X = trans_ratings.drop('rating', axis=1, inplace=False)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
X_tr = tf.convert_to_tensor(X_tr.to_numpy(), dtype=tf.float64)
X_te = tf.convert_to_tensor(X_te.to_numpy(), dtype=tf.float64)
y_tr = tf.convert_to_tensor(y_tr, dtype=tf.float64)
y_te = tf.convert_to_tensor(y_te, dtype=tf.float64)

### tensorflow   FatrixMachine
import tensorflow as tf
n,p = X_tr.shape
#k = 10

class Model(object):
    def __init__(self, wsize):
        w_init = tf.random_normal_initializer()
        self.W = tf.Variable(initial_value=w_init(shape=(wsize, 1),
                                                  dtype='float64'), trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(1,),
                                                  dtype='float64'), trainable=True)
        #self.W = tf.random.normal([wsize, 1], dtype="float64")
        #self.b = tf.random.normal([1], dtype="float64")

    def __call__(self, x):
        return tf.matmul(x, self.W) + self.b

model = Model(p)

def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y, desired_y))

def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)
    dW, db = t.gradient(current_loss, [model.W, model.b])
    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)

Ws, bs = [], []
losses = []
epochs = (100)
for epoch in range(epochs):
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    current_loss = loss(model(X_tr), y_tr)
    losses.append(current_loss)

    train(model, X_tr, y_tr, learning_rate=0.01)
    #print('Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f' % (epoch, Ws[-1], bs[-1], current_loss))
    print('Epoch %2d:  loss=%2.5f' % (epoch,  current_loss))

prediction = model(X_te[:10])
print('predictions')
print(prediction)
print('desired')
print(y_te[:10])

# 显示所有
plt.plot(range(epochs), losses, 'r',)
plt.show()

print(loss(model(X_te), y_te))
#test_accuracy = tf.keras.metrics.Accuracy()
#predictions= model(X_te)
#test_accuracy(predictions, y_te)
#print(test_accuracy.result())
