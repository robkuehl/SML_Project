import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.datasets import cifar10, mnist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from relevance_propagation.rel_prop_functions import *


def main():

    # set parameters
    load_saved_model = True
    model_type = 'dense'
    data_switch = 0  # 0: Cifar10
                     # 1: MNIST
    eps = 100
    beta = None         #Choose either eps or beta

    # get data
    train_images, train_labels, test_images, test_labels = get_data(data_switch)

    # load saved model
    if load_saved_model:
        model = tf.keras.models.load_model('./models/rel_prop_model_cifar.h5')

    # build, train and save model
    else:
        if data_switch:
            model = get_model(model_type=model_type, input_shape=(28,28))
        else:
            model = get_model(model_type=model_type, input_shape=(32,32,3))

        model.summary()

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['acc'])

        model = fit_model(model=model,
                          epochs=1,
                          batch_size=1000,
                          train_images=train_images,
                          train_labels=train_labels,
                          test_images=test_images,
                          test_labels=test_labels)

        if not os.path.exists('./models/'):
            os.makedirs('./models/')
        model.save('./models/rel_prop_model.h5')

    weights = get_weights(model)

    # Hilfsmodel zum Extrahieren der Outputs des Hidden Layers
    extractor = tf.keras.Model(inputs=model.inputs,
                               outputs=[layer.output for layer in model.layers])

    plot_rel_prop(model=model,
                  extractor=extractor,
                  images=train_images,
                  input_labels=train_labels,
                  data_switch=data_switch,
                  weights=weights,
                  eps=eps,
                  beta=beta)
    

if __name__ == "__main__":
    main()