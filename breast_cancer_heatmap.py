import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.activations import softmax, relu
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset = load_breast_cancer()
features = list(dataset['feature_names'])

X_vals = dataset['data']
y_vals = dataset['target']

X_train, X_test, y_train, y_test = train_test_split(X_vals, y_vals, random_state=42)

X_train_sc = StandardScaler().fit_transform(X_train, y_train)


model = Sequential([
    Dense(100, input_shape=(30,), activation='relu', use_bias=True),
    Dense(1, activation='sigmoid')
])

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=Adam(),
              metrics=['acc'])

model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=10
)

model.evaluate(X_test, y_test, verbose=2)

weights = model.weights[0].numpy()

bias_weights = np.reshape(model.weights[1].numpy(), [1,100])

weights = np.concatenate([weights, bias_weights])

plt.figure(figsize=(20, 5))
plt.imshow(weights, interpolation='none', cmap='viridis')
plt.yticks(range(31), features + ['bias'])
plt.colorbar()
plt.show()