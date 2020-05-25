from multi_classifier import multi_classifier
from binary_classifier import binary_classifier
from tensorflow.keras.datasets import cifar10, mnist
from rel_prop_functions import plot_rel_prop
from rel_prop_functions import rel_prop
import tensorflow as tf
import numpy as np

def get_multi_cl(dataset, model_type):
    mc = multi_classifier(dataset=dataset, model_type=model_type)
    mc.fit_classifiers(epochs=10, batch_size=100)
    return mc
    
def get_binary_cl(data, dataset, model_type, class_nb, epochs=10, batch_size=100):
    
    cl = binary_classifier(model_type="dense", dataset=dataset, class_nb=class_nb)
    cl.set_data(data)
    cl.set_model()
    cl.fit_model(epochs, batch_size)

    print("Model Accuracy: {}".format(cl.evaluate(10)))
    print("Model Accuracy for images with label {} : {}".format(class_nb, cl.non_trivial_accuracy()))
    
    #cl.model.save('./binary_models/model_{}_{}_{}e_{}bs.h5'.format(dataset,model_type,epochs,batch_size))
    
    return cl
    

def main_binary():
    dataset = 'mnist'
    model_type = 'dense'
    class_nb = 5
    
    if dataset == 'mnist':
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    elif dataset == 'cifar10':
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        
    data = [train_images, train_labels, test_images, test_labels]
    
    cl = get_binary_cl(data=data, dataset=dataset, model_type=model_type, class_nb=class_nb)
    model = cl.model
    
    
    # Führe Relevance Propagation für die ersten 10 Bilder der Klasse nb_class aus, die der Classifier korrekt erkennt
    j=0
    i=0
    while j<10:
        if test_labels[i]==class_nb:
            if model.predict(np.array([test_images[i]]))[0][0]==1:
                j+=1
                plot_rel_prop(model, test_images[i], eps=None, beta=None)
                #print(test_labels[i])
        i+=1

def main_multi():
    dataset = 'mnist'
    model_type = 'dense'
    
    if dataset == 'mnist':
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    elif dataset == 'cifar10':
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        
    
    mc = get_multi_cl(dataset=dataset, model_type=model_type)
    
    prediction = mc.predict([test_images[0]])
    print("Multiclassifier Prediction: {}".format(prediction))
    print("Correct Label: {}".format(test_labels[0]))
    
    one_pred = [prediction[i] for i in len(prediction) if prediction[0]==1]
    for element in one_pred:
        model = mc.classifiers[element[1]]
        plot_rel_prop(model, test_images[0], eps=None, beta=None)
    
    
    
if __name__ == '__main__':
    #main_binary()
    main_multi()