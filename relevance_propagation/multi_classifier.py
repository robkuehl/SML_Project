import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10, mnist

from binary_classifier import binary_classifier

class multi_classifier:
    
    def __init__(self, dataset, model_type):
        self.dataset = dataset
        self.model_type = model_type
        self.set_data()
        self.create_binary_classifiers()
        
    def set_data(self):
        if self.dataset == 'mnist':
            (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        elif self.dataset == 'cifar10':
            (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

        train_images = train_images / 255.
        test_images = test_images / 255.
        
        self.data = [train_images, train_labels, test_images, test_labels]
        
        
    def create_binary_classifiers(self):
        classes = set(list(self.data[1]))
        self.classifiers = [binary_classifier(model_type=self.model_type, dataset=self.dataset, class_nb=c) for c in classes]
        for cl in self.classifiers:
            cl.set_data(self.data)
            cl.set_model()
            
    def fit_classifiers(self, epochs, batch_size):
        for cl in self.classifiers:
            cl.fit_model(epochs, batch_size)
            
    def get_classifiers(self):
        return self.classifiers
        

    def predict(self, image):
        prediction = [(cl.predict(image), cl.class_nb) for cl in self.classifiers]
        return prediction

