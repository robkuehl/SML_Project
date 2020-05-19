import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.activations import softmax, relu
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images = train_images / 255.
test_images = test_images / 255.
# X_train_sc = StandardScaler().fit_transform(X_train, y_train)

# TODO: Scaling zu SandardScaler ändern.. hilft vielleicht

# model = Sequential([
#     Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
#     MaxPooling2D(pool_size=(2, 2)),
#     Flatten(),
#     Dense(10, activation='softmax')
# ])

# model = Sequential([
#     Flatten(input_shape= (32,32,3)),
#     Dense(4096, activation='relu', use_bias = False),
#     Dense(10, activation='softmax', use_bias = False)
# ])

# model.summary()

# model.compile(loss='sparse_categorical_crossentropy',
#               optimizer=Adam(),
#               metrics=['acc'])

# model.fit(
#     train_images,
#     train_labels,
#     epochs=10,
#     batch_size=1000,
#     validation_data=(test_images, test_labels)
# )

# model.save('C:/Users/marcp/models/rel_prop_model.h5')

model = tf.keras.models.load_model('C:/Users/marcp/models/rel_prop_model.h5')

first_weights = model.weights[0].numpy()
print(first_weights.shape)
second_weights = model.weights[1].numpy()
print(second_weights.shape)

#model.evaluate(test_images, test_labels)

# Hilfsmodel zum Extrahieren der Outputs des Hidden Layers
extractor = tf.keras.Model(inputs=model.inputs,
                        outputs=[layer.output for layer in model.layers])


# Funktion für Relevance Propagation
def rel_prop(input: np.ndarray) -> np.ndarray:
    features = extractor(np.array(np.array([input])))

    flattened_input = features[0].numpy()
    hidden_output = features[1].numpy()
    output = features[2].numpy()

    print("flattened_input shape {} ".format(flattened_input.shape))
    print("hidden_output shape {} ".format(hidden_output.shape))
    print("output shape {} ".format(output.shape))

    # Berechnung von R1
    R2 = np.transpose(output)

    nominator = np.multiply(np.transpose(hidden_output),
                            second_weights)

    print("shape nominator {}".format(nominator.shape))

    denominator = np.matmul(hidden_output,
                            second_weights)
    print("shape denominator {}".format(nominator.shape))

    fraction = np.divide(nominator, denominator)
    print("shape fraction {}".format(fraction.shape))
    R1 = np.dot(fraction, R2)

    # Berechnung von R0
    nominator = np.multiply(np.transpose(R1),
                            first_weights)

    denominator = np.matmul(flattened_input,
                            first_weights)

    fraction = np.divide(nominator, denominator)
    R0 = np.dot(fraction, R1)

    relevance = np.reshape(R0,(32,32,3))

    return relevance


# Test und Visualisierung
for i in range(0,5):
    image = train_images[i+40]
    test = rel_prop(np.array([image]))
    #test = np.sum(test, axis=2)
    reds_map = test[:,:,0]
    greens_map = test[:,:,1]
    blues_map = test[:,:,2]


    plt.subplot(4,5,i+1)
    plt.imshow(image)
    plt.subplot(4,5,i+6)
    plt.imshow(reds_map, cmap='cividis')
    plt.subplot(4,5,i+11)
    plt.imshow(greens_map, cmap='cividis')
    plt.subplot(4,5,i+16)
    plt.imshow(blues_map, cmap='cividis')

plt.show()