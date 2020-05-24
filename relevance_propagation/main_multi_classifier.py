from multi_classifier import multi_classifier
from binary_classifier import binary_classifier
from tensorflow.keras.datasets import cifar10, mnist
from rel_prop_functions import plot_rel_prop
from rel_prop_functions import rel_prop
import tensorflow as tf

def run_multi():
    mc = multi_classifier(which_dataset='mnist', model_type='dense')
    mc.fit_classifiers()
    print(mc.predict(mc.data[2][1]))
    
def run_binary(dataset, model_type, class_nb):
    if dataset == 'mnist':
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    elif dataset == 'cifar10':
        (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
        
    data = [train_images, train_labels, test_images, test_labels]
    
    cl = binary_classifier(model_type="dense", data_set=dataset, class_nb=class_nb)
    cl.set_data(data)
    cl.set_model()
    epochs = 10
    batch_size = 100
    cl.fit_model(epochs, batch_size)

    print("Model Accuracy: {}".format(cl.evaluate(10)))
    print("Model Accuracy for images with label {} : {}".format(class_nb, cl.non_trivial_accuracy()))
    
    model = cl.model
    model.save('./binary_models/model_{}_{}_{}e_{}bs.h5'.format(dataset,model_type,epochs,batch_size))
    relevances = []
    for i in range(100):
        plot_rel_prop(model,test_images[i], eps=None, beta=None)
        relevances.append(rel_prop(model, test_images[i]))
    
    for i in range(len(relevances)):
        print(test_labels[i], (relevances[i]!=0).any())
        
    return relevances
    

    
    
if __name__ == '__main__':
    relevances = run_binary(dataset='mnist', model_type='dense', class_nb=5)
    