# -*- coding: utf-8 -*-
# author: Austin
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7, 7)
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils


def print_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print(X_train.shape, X_test.shape, type(X_train))
    print(y_train.shape, y_test.shape)
    print(X_train[1, :, :])
    print(y_train[1])
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_train[i], cmap='gray', interpolation='none')
        plt.title("Class {}".format(y_train[i]))
    plt.show()


def bnn_model():
    nb_classes = 10
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = Sequential()

    model.add(Dense(512, input_shape=(784, )))
    model.add(Activation('relu'))
    model.add(Dropout(0, 2))

    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0, 2))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.fit(X_train, Y_train, batch_size=128, nb_epoch=4, verbose=1, validation_data=(X_test, Y_test))

    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score)
    predicted_classes = model.predict_classes(X_test)
    correct_indices = np.nonzero(predicted_classes == y_test)[0]
    incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
    print(correct_indices, incorrect_indices)

    plt.figure()
    for i, correct in enumerate(correct_indices[:9]):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_test[correct].reshape(28, 28), cmap='gray', interpolation='none')
        plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))

    plt.figure()
    for i, incorrect in enumerate(incorrect_indices[:9]):
        plt.subplot(3, 3, i + 1)
        plt.imshow(X_test[incorrect].reshape(28, 28), cmap='gray', interpolation='none')
        plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))
    plt.show()


if __name__ == '__main__':

    bnn_model()
