import tensorflow as tf
from tensorflow import keras
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import pandas as pd


def lossF(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def accuracyF(y_true, y_pred):
    threshold = 0.1
    x = tf.reduce_mean(tf.square(y_true - y_pred) / tf.square(tf.maximum(y_true, y_pred)))
    return x < threshold


path_train_in = r'/home/yxchen/Downloads/train_in.xlsx'
path_train_out = r'/home/yxchen/Downloads/train_out.xlsx'
path_test_in = r'/home/yxchen/Downloads/test_in.xlsx'
path_test_out = r'/home/yxchen/Downloads/test_out.xlsx'

train_data_in = pd.read_excel(path_train_in).values
train_data_out = pd.read_excel(path_train_out).values

test_data_in = pd.read_excel(path_test_in).values
test_data_out = pd.read_excel(path_test_out).values


# train_data_in = np.concatenate((train_data_in, test_data_in), axis=0)
# train_data_out = np.concatenate((train_data_out, test_data_out), axis=0)


# train_data_out = np.ravel(train_data_out)
# test_data_out = np.ravel(test_data_out)
#
# train_dataset = tf.data.Dataset.from_tensor_slices((train_data_in, train_data_out))
# train_dataset = train_dataset.shuffle(train_data_in.shape[0]).batch(1)
# test_dataset = tf.data.Dataset.from_tensor_slices((test_data_in, test_data_out))
# test_dataset = test_dataset.shuffle(test_data_in.shape[0]).batch(1)


# model = keras.Sequential([
#     keras.layers.Dense(5, activation='relu'),
#     keras.layers.Dense(3, activation='relu'),
#     keras.layers.Dense(1, activation='sigmoid')])
#
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# history = model.fit(train_dataset,
#                     epochs=30)

# loss, accuracy = model.evaluate(test_dataset)

class MyModel(keras.Model):

    def __init__(self):
        super(MyModel, self).__init__(name='my_model')
        # Define your layers here.
        self.dense_1 = keras.layers.Dense(5, activation=None,
                                          kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.01, seed=1),
                                          bias_initializer=tf.constant_initializer(0))
        self.dense_2 = keras.layers.Dense(3, activation=None,
                                          kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.01, seed=2),
                                          bias_initializer=tf.constant_initializer(0))
        self.dense_3 = keras.layers.Dense(1, activation=None,
                                          kernel_initializer=keras.initializers.TruncatedNormal(stddev=0.01, seed=3),
                                          bias_initializer=tf.constant_initializer(0))

    def call(self, inputs):
        # Define your forward pass here,
        # using layers you previously defined (in `__init__`).
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        x = self.dense_3(x)
        return x


avr = []
t_num = 200
for j in range(1, t_num):
    model = MyModel()
    model.compile(optimizer='adam',
                  loss=lossF,
                  metrics=[accuracyF])
    model.fit(train_data_in, train_data_out, epochs=j, batch_size=1, validation_data=(test_data_in, test_data_out))
    # model.fit(train_dataset, epochs=15)
    # loss, accuracy = model.evaluate(test_dataset)

    for i in range(test_data_in.shape[0]):
        data = (np.expand_dims(test_data_in[i], 0))
        p = model.predict(data)
        r = test_data_out[i]
        print('p: ' + str(p) + '  r: ' + str(r))

    s = 0
    g = []
    for i in range(test_data_in.shape[0]):
        data = (np.expand_dims(test_data_in[i], 0))
        p = model.predict(data)
        r = test_data_out[i]
        s += np.abs(p[0][0] - r[0]) / np.maximum(p[0][0], r[0])
        g.append(np.abs(p[0][0] - r[0]) / np.maximum(p[0][0], r[0]))
    avr.append(s / test_data_in.shape[0])
    # m = max(g)
    # print(avr, m)

x = list(range(1, t_num))
y = [a for a in avr]
plt.scatter(x, y, s=20)
plt.axis([0, t_num, 0, 2])
plt.show()
