# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 18:40:44 2020

@author: groes
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch


def split_data(X, y, test_size, normalise=True):
    """
    Splits and normalizes data
    
    """
    
    X_train_proto, X_test, y_train_proto, y_test = train_test_split(X, y, test_size = test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_train_proto, y_train_proto, test_size = test_size)
    y_train_onehot = one_hot_encoding(y_train)
    y_test_onehot = one_hot_encoding(y_test)        
    y_val_onehot = one_hot_encoding(y_val) 
    
    if normalise:
        X_train = normalize(X_train)
        X_test = normalize(X_test)
        X_val = normalize(X_val)
        #X_train = np.apply_along_axis(normalise_input, 1, X_train)
        #X_test = np.apply_along_axis(normalise_input, 1, X_test)
    
    return X_train, X_test, X_val, y_train_onehot, y_test_onehot, y_val_onehot
    

    

def normalise_input(datapoint):
    Mu = sum(datapoint)/len(datapoint)
    SD = sum((datapoint-Mu)*(datapoint-Mu))/len(datapoint)
    znorm = (datapoint - Mu)/np.sqrt(SD + 0.0001)
    return znorm


def one_hot_encoding(data):
    onehot = [] 
    vector_length = len(np.unique(data))
    for i in data:
        vector = np.zeros(vector_length)
        vector[int(i)] = 1
        onehot.append(vector)
         
    return np.array(onehot)

def standardise(X):
        return X/np.max(X)

def generate_confusion_matrix(all_predicted, all_labels):
    cat_all_predicted = torch.cat((all_predicted[0], all_predicted[1]))
    for tensor in all_predicted[2:]:
        cat_all_predicted = torch.cat((cat_all_predicted, tensor))
    
    cat_all_labels = torch.cat((all_labels[0], all_labels[1]))
    for tensor in all_labels[2:]:
        cat_all_labels = torch.cat((cat_all_labels, tensor))

    confmatrix = confusion_matrix(cat_all_predicted, cat_all_labels) 
    return confmatrix, cat_all_predicted, cat_all_labels



def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    # Source: https://deeplizard.com/learn/video/0LhiS6yu2qQ 
    import itertools

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
