import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
#from tensorflow.keras.layers import ReLU
from tensorflow.keras.optimizers import Adam
#from sklearn.preprocessing import StandardScaler
#from tensorflow.keras.activations import softmax, relu
from tensorflow.keras.datasets import cifar10, mnist
#from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class binary_classifier:
    
    def __init__(self, model_type, data_set, class_nb):
        self.model_type = model_type
        assert(type(self.model_type)==str)
        self.data_set = data_set
        self.class_nb = class_nb
        

    def set_data(self, data):
        self.train_images = data[0]
        self.train_labels =  data[1]
        self.test_images = data[2]
        self.test_labels = data[3]
        
        #make_binary_data
        self.train_labels = (self.train_labels==self.class_nb).astype(int)*self.class_nb
        self.test_labels = (self.test_labels==self.class_nb).astype(int)*self.class_nb

    def set_model(self):
        
        if self.data_set == 'mnist':
            input_shape=(28,28)
        else:
            input_shape=(32,32,3)
        
        if self.model_type == "dense":
            model = Sequential([
                Flatten(input_shape=input_shape),
                Dense(4096, activation='relu', use_bias=False),
                Dense(2, activation='softmax', use_bias=False)
            ])

        model.summary()

        model.compile(loss='sparse_categorical_crossentropy',
                    optimizer=Adam(),
                    metrics=['acc'])

        self.model = model


    def fit_model(self, epochs: int, batch_size: int):
        with tf.device("/gpu:0"):
            self.model.fit(
                self.train_images,
                self.train_labels,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(self.test_images, self.test_labels)
            )

    def predict(self, image):
        pred = self.model.predict(image)
        return pred