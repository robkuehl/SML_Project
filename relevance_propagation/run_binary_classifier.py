#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 11:46:37 2020

@author: robin
"""



from tensorflow.keras.datasets import cifar10, mnist
dataset = mnist
(train_images, train_labels), (test_images, test_labels) = dataset.load_data()
new_labels = (test_labels==1).astype(int)
data = [train_images, train_labels, test_images, test_labels]
from binary_classifier import binary_classifier
cl = binary_classifier(model_type="dense", data_set='mnist', class_nb=5)
cl.set_data(data)
cl.set_model()
cl.fit_model(10,10)

answers = []
for i in range(len(list(test_labels))):
    if test_labels[i]==5:
        answers.append(int(cl.predict(test_images[i].reshape(28,28))[0][0]))
        
print(sum(answers)/len(answers))
    
labels = cl.train_labels

'''
import matplotlib.pyplot as plt
# pick a sample to plot
sample = 0
image = test_images[sample]# plot the sample
fig = plt.figure
plt.imshow(image, cmap='gray')
plt.show()

print(test_labels[0])
'''