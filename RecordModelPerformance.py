# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 19:15:23 2020

@author: groes
"""
import sys

class RecordModelPerformance():
    def __init__(self, model_object, result_dict, model_name, epochs,
                 batch_size, learning_rate):
        """
        

        Parameters
        ----------
        model_object : class instance
            Instance of the neural network class that you want to run and record
            the results of
        result_dict : DICTIONARY
            A dictionary to which results can be appended, ie. its values should be lists.
            If non-empty, the dictionary should have the following keys as strings:
                model_name ; epochs ; batch_size ; learning_rate ; accuracy ; 
                criterion ; optimizer ; model_architecture
        model_name : STRING
            Some descriptive name of the model, e.g. "ConvNet1"
        epochs : INT
            Number of epochs
        batch_size : INT
            Number of datapoints in each batch
        learning_rate : FLOAT
            The model's learning rate

        Returns
        -------
        None.

        """
        
        self.model_object = model_object
        self.result_dict = result_dict
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
    def run(self):
        """
        This method is designed to make it easier to run several models and compare them.
        
        The idea is that it is called iteratively with different parameters each time, 
        e.g. you make a list with n model names, epochs, batch_sizes etc and loop over them,
        calling this method in every loop.

        Parameters
        ----------
        NONE (it uses the parameters given to __init__)

        Returns
        -------
        A dictionary with most relevant model parameters

        """
        

        if len(self.result_dict) == 0:
            self.result_dict = {"model_name" : [self.model_name], "epochs" : [self.epochs],
                           "batch_size" : [self.batch_size], "learning_rate" : [self.learning_rate],
                           "accuracy" : [], "criterion" : [], "optimizer" : [], "model_architecture" : [] }
        else:
            for key in ["model_name", "epochs", "batch_size", "learning_rate", "accuracy", "criterion", "optimizer", "model_architecture"]:
                if key not in self.result_dict:
                    sys.exit("key " + key + " not in result_dict")
            self.result_dict["model_name"].append(self.model_name)
            self.result_dict["epochs"].append(self.epochs)
            self.result_dict["batch_size"].append(self.batch_size)
            self.result_dict["learning_rate"].append(self.learning_rate)
        
        model = self.model_object
        model.train(num_epochs = self.epochs, batch_size = self.batch_size, learning_rate = self.learning_rate)
        test_accuracy = model.test()
        
        self.result_dict["accuracy"].append(test_accuracy)
        self.result_dict["criterion"].append(model.criterion)
        self.result_dict["optimizer"].append(model.optimizer)
        self.result_dict["model_architecture"].append(str(self.model_object))
        
        return self.result_dict
