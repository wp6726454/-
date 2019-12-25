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


model = MyModel()
model.compile(optimizer='adam',
              loss=lossF,
              metrics=[accuracyF])

checkpoint_path = 'cp.ckpt'
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

model.fit(train_data_in, train_data_out, epochs=10, batch_size=1, validation_data=(test_data_in, test_data_out),
          callbacks=[cp_callback])

new_model = MyModel()
new_model.compile(optimizer='adam',
                  loss=lossF,
                  metrics=[accuracyF])
new_model.load_weights(checkpoint_path)
