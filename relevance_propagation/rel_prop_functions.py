import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.activations import softmax, relu
from tensorflow.keras.datasets import cifar10, mnist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def get_data(data_switch) -> (np.ndarray, np.ndarray):
    if data_switch:
        dataset = mnist
    else:
        dataset = cifar10

    (train_images, train_labels), (test_images, test_labels) = dataset.load_data()
    train_images = train_images / 255.
    test_images = test_images / 255.

    return train_images, train_labels, test_images, test_labels


def get_model(model_type: str, input_shape: tuple) -> tf.keras.Sequential:
    if model_type == 'conv':
        model = Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(10, activation='softmax')
        ])

    elif model_type == "dense":
        model = Sequential([
            Flatten(input_shape=input_shape),
            Dense(4096, activation='relu', use_bias=False),
            Dense(10, activation='softmax', use_bias=False)
        ])

    model.summary()

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['acc'])

    return model


def fit_model(model: tf.keras.Sequential, epochs: int, batch_size: int, train_images: np.ndarray,
              train_labels: np.ndarray, test_images: np.ndarray, test_labels: np.ndarray):
    with tf.device("/cpu:0"):
        model.fit(
            train_images,
            train_labels,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(test_images, test_labels)
        )

    return model


def get_weights(model: tf.keras.Sequential) -> (np.ndarray, np.ndarray):
    #TODO return array for variable number of layers

    first_weights = model.weights[0].numpy()
    second_weights = model.weights[1].numpy()

    return first_weights, second_weights


# Funktion für Relevance Propagation
def rel_prop(model: tf.keras.Sequential, input: np.ndarray, eps: float = 0, beta: float = None) -> np.ndarray:
    first_weights, second_weights = get_weights(model)

    # Hilfsmodel zum Extrahieren der Outputs des Hidden Layers
    extractor = tf.keras.Model(inputs=model.inputs,
                               outputs=[layer.output for layer in model.layers])

    features = extractor(np.array([input]))

    flattened_input = features[0].numpy()
    hidden_output = features[1].numpy()
    output = features[2].numpy()

    # Berechnung von R1
    r2 = np.transpose(output)

    r1 = calc_r(r=r2,
                output=hidden_output,
                weights=second_weights,
                eps=eps,
                beta=beta)

    r0 = calc_r(r=r1,
                output=flattened_input,
                weights=first_weights,
                eps=eps,
                beta=beta)

    relevance = np.reshape(r0, input.shape)

    return relevance


def calc_r(r: np.ndarray, output: np.ndarray, weights: np.ndarray, eps: int = 0, beta: int = None):

    nominator = np.multiply(np.transpose(output),
                            weights)

    if beta is not None:
        if eps:
            print('+++ERROR+++')
            print('Choose either EPS or BETA, not both!')
            print('+++ERROR+++')
            sys.exit()

        zero = np.zeros(nominator.shape)
        z_pos = np.maximum(zero, nominator)
        z_neg = np.minimum(zero, nominator)

        denominator_pos = np.sum(z_pos, axis=0)
        denominator_neg = np.sum(z_neg, axis=0)

        fraction_pos = np.divide(z_pos, denominator_pos)
        fraction_neg = np.divide(z_neg, denominator_neg)

        fraction = (1 - beta) * fraction_pos + beta * fraction_neg

    else:
        denominator = np.matmul(output,
                                weights)

        if eps:
            denominator = denominator + eps * np.sign(denominator)

        fraction = np.divide(nominator, denominator)

    r_new = np.dot(fraction, r)

    return r_new

def plot_rel_prop(model: tf.keras.Sequential, image: np.ndarray, eps: float, beta: float):
    
    # Test und Visualisierung


    prop = rel_prop(model, image)
    
    # bei MNIST auskommentieren
    #prop = np.sum(prop, axis=2)

    plt.subplot(2,1,1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(prop, cmap='cividis')

    plt.subplot(2,1,2)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    
    
    prop2 = rel_prop(model, image, eps=eps, beta=beta)
    
    # bei MNIST auskommentieren
    #prop2 = np.sum(prop2, axis=2)
    plt.imshow(prop2, cmap='cividis')

    plt.show()