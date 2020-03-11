import os
from Morph_dil import dilate
from Morph_ero import Erossion
import keras

import matplotlib.pyplot as plt
import numpy as np
import pickle

from keras.layers import Input, Conv2D, concatenate, \
    Dropout, Dense, MaxPooling2D, Flatten, subtract
from keras.layers import Activation
from keras.models import Model
import keras.backend as K
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import np_utils

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

batch_size = 25
height, width = 28, 28
nb_classes = 10

### data
def get_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    x_train = x_train.reshape(x_train.shape[0], height, width, 1)
    x_test = x_test.reshape(x_test.shape[0], height, width, 1)

    #for i in range(0,x_test.shape[0]):
    #    x_test[i] = 1 - x_test[i]
        # plt.imshow(x_test[i].reshape(height, width), cmap='gray'); plt.show(); break

    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    return x_train, y_train, x_test, y_test

from keras.utils import plot_model
### model
def morphnet(in_mnist=(height, width, 1), scale=1):
    img = Input(shape=in_mnist, name='mnist')
    
    x1 = Erossion(filters = 16, kernel_size = (3,3), strides = (1,1), operation='e')(img)
    x1 = Activation('relu')(x1)
    x1 = Erossion(filters = 32, kernel_size = (3,3), strides=(1, 1), operation='e')(x1)
    x1 = Activation('relu')(x1)
    
    x2 = dilate(filters = 16,kernel_size = (3,3), strides = (1,1), operation='d')(img)
    x2 = Activation('relu')(x2)
    x2 = dilate(filters = 32, kernel_size = (3,3), strides = (1,1), operation='d')(x2)
    x2 = Activation('relu')(x2)
    
    res = subtract([x2,x1])
    res = MaxPooling2D(pool_size = (2,2), strides = (2,2))(res)
    res = MaxPooling2D(pool_size = (2,2), strides = (2,2))(res)
    m = Flatten()(res)
    m = Dense(units=512,activation='relu')(m)
    m = Dropout(0.5)(m)

    out = Dense(units=10, activation='softmax')(m)

    model = Model(inputs=img,outputs=out)
    opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    plot_model(model, to_file='model.png')
  
    return model

### train
def train(epoch=1):
    x_train, y_train, x_test, y_test = get_data()

    model = morphnet()

    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch, verbose=1) #validation_data=(x_test, y_test)

    #plt.figure(num='accuracy vs epoch')
    #plt.plot(history.history['acc'])
    #plt.plot(history.history['loss']);
    #plt.ylabel('accuracy / loss'); plt.xlabel('epoch'); plt.legend(['acc', 'loss'], loc='upper right')
    #plt.show()

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

if __name__ == '__main__':
    print('..............')
    train(epoch=60)
