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


# nLayers, neuronsPerLayer and activationFunc can now be set different
def get_model(model_type: str, input_shape: tuple, nLayers = 1, neuronsPerLayer = 4096, activationFunc = 'relu') -> tf.keras.Sequential:
    if model_type == 'conv':
        model = Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(10, activation='softmax')
        ])

    elif model_type == "dense":
        model = Sequential([
            Flatten(input_shape=input_shape)
        ])
        for layer in range(nLayers):
            model.add(Dense(neuronsPerLayer, activation=activationFunc, use_bias=False))
        model.add(Dense(10, activation='softmax', use_bias=False))

    model.summary()

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(),
                  metrics=['acc'])

    return model

# Fit model method, variable devices. 'marc_gpu' skips tf.device(...) part to avoid errors...
def fit_model(model: tf.keras.Sequential, epochs: int, batch_size: int, train_images: np.ndarray,
              train_labels: np.ndarray, test_images: np.ndarray, test_labels: np.ndarray, device = 'cpu'):
    if device == 'cpu':
        with tf.device("/cpu:0"):
            model.fit(
                train_images,
                train_labels,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(test_images, test_labels)
            )
    elif device == 'gpu':
        with tf.device("/gpu:0"):
            model.fit(
                train_images,
                train_labels,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(test_images, test_labels)
            )
    elif device == 'gpu_marc':
        model.fit(
                train_images,
                train_labels,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(test_images, test_labels)
            )
    return model


# Funktion für Relevance Propagation NEU (beliebige anzahl schichten)
def rel_prop_multilayer(model: tf.keras.Sequential, input: np.ndarray, eps: float = 0.0, beta: float = None) -> np.ndarray:

    # Hilfsmodel zum Extrahieren der Outputs des Hidden Layers
    extractor = tf.keras.Model(inputs=model.inputs,
                               outputs=[layer.output for layer in model.layers])

    features = extractor(np.array([input]))
    #extract number of features
    nFeatures = len(features)
    outputs = []
    weightMatrices = []
    for layer in range(nFeatures):
        outputs.append(features[layer].numpy())
    # decrease nFeatures to collect weight matrices (one less than no. of features)
    nFeatures-=1
    weights = []
    for betweenLayer in range(nFeatures):
        weightMatrices.append(model.weights[betweenLayer].numpy())
    # Equivalent to R^{(l+1)}, initiated with the output
    rel_prop_vector_2 = np.transpose(outputs[nFeatures])
    # again, decrease nFeatures by one to get access to the list indices -> Backwards calculation of input Relevance vector
    nFeatures-=1
    while(nFeatures>=0):
        #rel_prop_vector_1 is equivalent to R^{(l)} from the notebook
        rel_prop_vector_1= calc_r(r=rel_prop_vector_2, 
                output=outputs[nFeatures],
                weights=weightMatrices[nFeatures],
                eps=eps,
                beta=beta)
        nFeatures-=1
        rel_prop_vector_2 = rel_prop_vector_1

    # Finally, output of the relevance propagation is the same dimension as the flattened input vector, 
    # reshape it to original dimensions
    relevance = np.reshape(rel_prop_vector_1, input.shape)

    return relevance


# calculate the next relevance propagation vector, according to notebook formula
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


    prop = rel_prop_multilayer(model, image)
    
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
    
    
    prop2 = rel_prop_multilayer(model, image, eps=eps, beta=beta)
    
    # bei MNIST auskommentieren
    #prop2 = np.sum(prop2, axis=2)
    plt.imshow(prop2, cmap='cividis')

    plt.show()


# %%%%%%%%%%% ALTER KRAM - EVTL WEG? %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Wird momentan noch im Code verwendet, ggf. anpassen

#############################################################################

def get_weights(model: tf.keras.Sequential) -> (np.ndarray, np.ndarray):
    #TODO return array for variable number of layers

    first_weights = model.weights[0].numpy()
    second_weights = model.weights[1].numpy()

    return first_weights, second_weights


# Funktion für Relevance Propagation - Hardcode Version, nur 1 Dense Layer
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

def plot_value_array(predictions_array, true_label):
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


## Plot relevance propagation for multiclassifier...
def plot_rel_prop(model: tf.keras.Sequential,extractor: tf.keras.Model, images: np.ndarray,
                  input_labels: np.ndarray, data_switch: int, weights: tuple, eps: float, beta: float):
    if data_switch==1:
        labels = [i for i in range(0, 10)]
    else:
        labels = {0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer',
                  5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'}

    # Test und Visualisierung
    for i in range(0,5):
        idx = i+200
        image = images[idx]
        test1 = rel_prop_multilayer(model, image)
        if not data_switch:
            test1 = np.sum(test1, axis=2)
            label = labels[input_labels[idx][0]]
        else:
            label = labels[input_labels[idx]]

        plt.subplot(4,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.xlabel(label)
        plt.imshow(image)
        plt.subplot(4,5,i+6)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test1, cmap='cividis')
        test = rel_prop_multilayer(model, image, eps=eps, beta=beta)
        if not data_switch:
            test = np.sum(test, axis=2)
        plt.subplot(4,5,i+11)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.xlabel('eps={eps}, beta={beta}')
        plt.imshow(test, cmap='cividis')
        plt.subplot(4,5,i+16)
        pred = model.predict(np.array([image]))[0]
        if not data_switch:
            plot_value_array(pred, input_labels[idx][0])
        else:
            plot_value_array(pred, input_labels[idx])

    plt.show()