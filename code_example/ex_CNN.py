# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 2023
@author: subinJoo
reference : https://keras.io/examples/vision/mnist_convnet/
"""
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import random
# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# Load the data and split it between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

from tensorflow.keras.optimizers import Adam
optimizer = Adam(learning_rate=0.001, clipnorm=0.1)

# create training function
def training(inputs,outputs):
    model_params = model.trainable_variables
    with tf.GradientTape() as tape:
        predicts = model(inputs)
        loss = tf.keras.metrics.categorical_crossentropy(predicts,outputs)
        
    # 오류함수를 줄이는 방향으로 모델 업데이트
    grads = tape.gradient(loss, model_params)
    optimizer.apply_gradients(zip(grads, model_params))
    return np.mean(loss)

    
""" training part
"""
batch_size = 128
epoch = 10000
list_loss = []
for i in range(epoch):
    batch = random.sample(list(range(len(x_train))),batch_size)    
    inputs = x_train[batch,:,:,:]
    outputs = y_train[batch,:]
    loss = training(inputs,outputs)
    if(i%10 == 0):
        print('epoch ' ,i,' / loss : ',loss)
        list_loss.append(loss)

# plot loss value per epoch    
import matplotlib.pyplot as plt
plt.plot(list_loss)

""" validation part
"""
batch = random.sample(list(range(len(x_test))),batch_size)    
inputs = x_test[batch,:,:,:]
outputs = y_test[batch,:]
predicts_test = model(inputs)
    
acc = np.sum(tf.math.argmax(predicts_test,axis=1) == tf.math.argmax(outputs,axis=1))/batch_size * 100
print('validation accuracy (%) : ',acc)

