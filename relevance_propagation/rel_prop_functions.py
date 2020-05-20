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
def rel_prop(model: tf.keras.Sequential, extractor: tf.keras.Model, input: np.ndarray, weights: tuple) -> np.ndarray:
    first_weights, second_weights = weights

    features = extractor(np.array([input]))

    flattened_input = features[0].numpy()
    hidden_output = features[1].numpy()
    output = features[2].numpy()

    # Berechnung von R1
    r2 = np.transpose(output)

    nominator = np.multiply(np.transpose(hidden_output),
                            second_weights)

    denominator = np.matmul(hidden_output,
                            second_weights)

    fraction = np.divide(nominator, denominator)
    r1 = np.dot(fraction, r2)

    # Berechnung von R0
    nominator = np.multiply(np.transpose(flattened_input),
                            first_weights)

    denominator = np.matmul(flattened_input,
                            first_weights)

    fraction = np.divide(nominator, denominator)
    r0 = np.dot(fraction, r1)

    relevance = np.reshape(r0, (28, 28))

    return relevance


def plot_value_array(predictions_array, true_label):
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def plot_rel_prop(model: tf.keras.Sequential,extractor: tf.keras.Model, images: np.ndarray,
                  input_labels: np.ndarray, data_switch: int, weights: tuple):
    if data_switch:
        labels = [i for i in range(0, 10)]
    else:
        labels = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer',
                  5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

    # Test und Visualisierung
    for i in range(0,5):
        idx = i+200
        image = images[idx]
        label = labels[input_labels[idx]]
        test = rel_prop(model, extractor, image, weights)
        if not data_switch:
            test = np.sum(test, axis=2)

        plt.subplot(3,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.xlabel(label)
        plt.imshow(image)
        plt.subplot(3,5,i+6)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test, cmap='cividis')
        plt.subplot(3,5,i+11)
        pred = model.predict(np.array([image]))[0]
        plot_value_array(pred, input_labels[idx])

    plt.show()