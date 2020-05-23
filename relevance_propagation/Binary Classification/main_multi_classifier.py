from multi_classifier import multi_classifier
from binary_classifier import binary_classifier
from tensorflow.keras.datasets import cifar10, mnist

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
    cl.fit_model(2,100)

    print("Model Accuracy: {}".format(cl.evaluate(10)))
    print("Model Accuracy for images with label {} : {}".format(class_nb, cl.non_trivial_accuracy()))

    
    
if __name__ == '__main__':
    run_binary(dataset='mnist', model_type='dense', class_nb=5)