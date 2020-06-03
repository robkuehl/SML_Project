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
from keras.utils import to_categorical

class Multilabel_Classifier():
    
    def __init__(self, dataset):
        self.dataset = dataset
        self.data = self.set_data()
    
    def set_data(self) -> (np.ndarray, np.ndarray):
        if self.dataset=='mnist':
            (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
            
        elif self.dataset=='cifar10':
            (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
    
        train_images = train_images / 255.
        test_images = test_images / 255.
        
        train_labels = to_categorical(train_labels)
        test_labels = to_categorical(test_labels)
    
        data = {"train_images": train_images, "train_labels": train_labels,
                "test_images": test_images, "test_labels": test_labels}
        
        return data
    
    
    def set_model(self, acti, loss) -> tf.keras.Sequential:
     
        model = Sequential([
            Flatten(input_shape=self.data['train_images'][0].shape),
            Dense(4096, activation='relu', use_bias=False),
            Dense(10, activation=acti, use_bias=False)
        ])
    
        model.summary()
    
        model.compile(loss=loss,
                      optimizer=Adam(),
                      metrics=['acc'])
    
        self.model = model
    
    
    def fit_model(self,epochs: int, batch_size: int):
    
            self.model.fit(
                self.data['train_images'],
                self.data['train_labels'],
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(self.data['test_images'], self.data['test_labels'])
            )
    

if __name__ == '__main__':
    mc = Multilabel_Classifier('mnist')
    loss = tf.keras.losses.Hinge()
    loss = 'binary_crossentropy'
    acti = 'tanh'
    acti = 'sigmoid'
    mc.set_model(acti,loss)
    mc.fit_model(10,100)
    mc.model.predict(np.array([mc.data['test_images'][0]]))
    mc.model.evaluate(mc.data['test_images'], mc.data['test_labels'])
