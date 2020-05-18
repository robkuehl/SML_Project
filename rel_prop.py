import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.activations import softmax, relu
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# X_train_sc = StandardScaler().fit_transform(X_train, y_train)


model = Sequential([
    Conv2D(filters=16, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(10, activation='softmax')
])

# model.summary()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(),
              metrics=['acc'])

model.fit(
    train_images,
    train_labels,
    epochs=10,
    batch_size=10
)

# model.evaluate(X_test, y_test, verbose=2)
#
# weights = model.weights[0].numpy()
#
# bias_weights = np.reshape(model.weights[1].numpy(), [1,100])
#
# weights = np.concatenate([weights, bias_weights])
#
# plt.figure(figsize=(20, 5))
# plt.imshow(weights, interpolation='none', cmap='viridis')
# plt.yticks(range(31), features + ['bias'])
# plt.colorbar()
# plt.show()