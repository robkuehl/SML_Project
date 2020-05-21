from multi_classifier import multi_classifier

def main_multi():
    mc = multi_classifier(which_dataset='mnist', model_type='dense')
    mc.fit_classifiers()
    print(mc.predict(mc.data[2][1]))
    
    
if __name__ == '__main__':
    main_multi()