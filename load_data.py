# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:49:50 2020

@author: groes
"""
from sklearn.datasets import fetch_openml

def load_data():
    '''
    

    Returns
    -------
    mnist : utils.Bunch
        DESCRIPTION.
        The mnist dataset

    '''
    
    # The next line is borrowed from here: https://github.com/ageron/handson-ml/issues/301
    mnist = fetch_openml("mnist_784", version=1, cache=True)
    
    return mnist