# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 21:04:42 2020

@author: hugha
"""
import numpy as np
import random
#import utils
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

class new_neural_network:
    def __init__(self, learning_rate):
        self.lr = learning_rate        
        self.layers = []


    # Adding methods for creating layers to make it easier to create layers 
    # with different parameters
    def create_input_layer(self, number_of_neurons, bias=0.0001):
        # Storing parameters in dictionary inspired by suggestion in here to keep
        # logs of settings stored in json for record keeping
        # https://stats.stackexchange.com/questions/352036/what-should-i-do-when-my-neural-network-doesnt-learn#:~:text=Too%20few%20neurons%20in%20a,%22memorize%22%20the%20training%20data
        parameters = {
            "layer_type" : "input",
            "bias" : bias,
            "number_of_neurons" : number_of_neurons,# making it parameterizable such that we can test the network on XOR
            "activations" : [],
            "bias_update" : [],
            "stored_bias" : []
            }
        self.layers.append(parameters)
        
    def add_hidden_layer(self, number_of_neurons, activation_func="relu", bias=0.0001):
        parameters = {
            "layer_type" : "hidden",
            "bias" : bias,
            "number_of_neurons" : number_of_neurons, 
            "activation_func" : activation_func,
            "weight_matrix" : self.initialize_weight_matrix(number_of_neurons),
            "activations" : [],
            "g_inputs" : [],
            "s" : 0,
            "r" : 0,
            "delta" : [],
            "weight_update" : [],
            "bias_update" : [],
            "stored_weights" : [],
            "stored_bias" : []
            }
        self.layers.append(parameters)

    def add_output_layer(self, number_of_neurons, activation_func="softmax"):
        parameters = {
            "layer_type" : "output",
            "number_of_neurons" : number_of_neurons, 
            "activation_func" : activation_func,
            "weight_matrix" : self.initialize_weight_matrix(number_of_neurons),
            "activations" : [],
            "g_inputs" : [],
            "s" : 0,
            "r" : 0,
            "delta" : [],
            "weight_update" : [],
            "stored_weights" : [],
            "stored_bias" : []
            }
        self.layers.append(parameters)

    def initialize_weight_matrix(self, number_of_neurons):
        return np.random.uniform(-1,1,(self.layers[-1]["number_of_neurons"],number_of_neurons))
        
        
    def new_train(self, X_train, X_val, y_train, y_val, epochs, batch_size = 128, optimiser = "Adam", tolerance = 0.01, max_patience = 10, stopping_criterion = "xent"):
        print("Training Started")
        stop_crit_prev = 1
        patience = 0
        epoch = 1
        
        val_xent_list = []
        train_xent_list = []
        val_accuracy_list = []
        train_accuracy_list =[]
        
        #epochs and no improvement stopper
        while epoch <= epochs and patience < max_patience:  
            mini_batches = self.get_minibatches(X_train,y_train, batch_size)           
            for batch in mini_batches:
                X_batch = [x for (x,y) in batch]
                y_batch = [y for (x,y) in batch]
                #print("forward pass started")
                self.forward_pass(X_batch)
                #print("backward pass started")
                self.backward_pass(y_batch, optimiser)
 
            #calculate accuracy on a portion of training data  
            train_batches = self.get_minibatches(X_train,y_train,round(len(X_train)*0.2) )
            train_X_batch = [x for (x,y) in train_batches[0]]
            train_y_batch = [y for (x,y) in train_batches[0]]
            
            #validation set accuracy
            val_accuracy = self.accuracy_score(X_val, y_val)
            val_accuracy_list.append(val_accuracy)
            
            #training set accuracy
            train_accuracy = self.accuracy_score(train_X_batch, train_y_batch)            
            train_accuracy_list.append(train_accuracy)
            
            #validate set cross entropy
            self.forward_pass(X_val)
            val_xent = self.xent(self.layers[-1]['activations'], y_val)
            val_xent_list.append(sum(val_xent)/len(val_xent))
            #training set cross entropy
            self.forward_pass(X_train)
            train_xent = self.xent(self.layers[-1]['activations'], y_train)
            train_xent_list.append(sum(train_xent)/len(train_xent))

            
            #gives option of stopping criterion
            if stopping_criterion == "xent":
                xent_mean = sum(val_xent)/len(val_xent)
                stop_crit = xent_mean
            elif stopping_criterion == "accuracy": 
                stop_crit = val_accuracy
            else:
                print("xent or accuracy, you choose")  
                
            if abs(stop_crit - stop_crit_prev)<tolerance:
                if patience == 0:
                    #stores weights before they overfit
                    for layer in self.layers:
                        if layer['layer_type'] == "input":
                            layer["stored_bias"] = layer["bias"]
                        elif layer['layer_type'] == "output":
                            layer["stored_weights"] = layer["weight_matrix"]
                        else:
                            layer["stored_bias"] = layer["bias"]
                            layer["stored_weights"] = layer["weight_matrix"]
                         
                        
                #returns matrices to their pre over-fitting peak           
                if patience == max_patience-1:
                    for layer in self.layers:
                        if layer['layer_type'] == "input":
                            layer["bias"] = layer["stored_bias"]
                        elif layer['layer_type'] == "output":
                            layer["weight_matrix"] = layer["stored_weights"]
                        else:
                            layer["bias"] = layer["stored_bias"]
                            layer["weight_matrix"] = layer["stored_weights"]                                   
                patience +=1
            else:
                patience = 0
                             
            stop_crit_prev = stop_crit 
             
            print("Epoch {}: loss {}, accuracy = {}".format(epoch,val_xent, val_accuracy))
            epoch +=1
        fig = plt.figure()
        fig.add_subplot(2,2,1)
        plt.plot(range(epoch-1), val_accuracy_list, label = 'validation')
        plt.plot(range(epoch-1), train_accuracy_list, label = 'training')
        plt.xlabel('Accuracy over epochs')
        
        fig.add_subplot(2,2,3)
        plt.plot(range(epoch-1), val_xent_list, label = 'validation')
        plt.plot(range(epoch-1), train_xent_list, label = 'training')
        plt.xlabel('Cross Entropy over epochs')
        fig.show()
        
    def get_minibatches(self, X, y, batch_size):
        training_data = [n for n in zip(X,y)]        
        random.shuffle(training_data)
        n = len(training_data)
        mini_batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
        return mini_batches
        
    
    def forward_pass(self, X):
        for layer in range(len(self.layers)):
            if self.layers[layer]["layer_type"] == "input":
                self.layers[layer]["activations"] = X                    
            else:
                self.layers[layer]['activations'], self.layers[layer]['g_inputs'] = self.calculate_activations(self.layers[layer-1], self.layers[layer])
         
    def backward_pass(self, y_batch, optimiser):
        #calculating the weight and bias updates
        for layer in range((len(self.layers))-1, -1, -1):
                    
            #output
            if self.layers[layer]["layer_type"] == "output":
                #include this in the line below
                self.layers[layer]['delta'] = (self.layers[layer]['activations'] - y_batch) #xent
                if optimiser == "SGD":
                    self.layers[layer]['weight_update'] = - self.lr * sum(self.weight_update((self.layers[layer]['activations'] - y_batch),  self.layers[layer-1]['activations']))     
                elif optimiser == "Adam":
                    self.layers[layer]['weight_update'] = - self.lr * self.Adam(self.layers[layer], self.layers[layer-1])
             #input      
            elif self.layers[layer]["layer_type"] == "input":
                self.layers[layer]['bias_update'] = - self.lr * sum(self.layers[layer+1]['delta'])  
            #hidden    
            else:
                self.layers[layer]['delta'] = self.cost_prime_hidden(self.layers[layer], self.layers[layer+1])                                        
                #get weight update
                if optimiser == "SGD":
                    self.layers[layer]['weight_update'] = - self.lr * self.SGD(self.layers[layer], self.layers[layer-1])
                elif optimiser == "Adam":
                    self.layers[layer]['weight_update'] = - self.lr * self.Adam(self.layers[layer], self.layers[layer-1])
                
                self.layers[layer]['bias_update'] = - self.lr *sum(self.layers[layer+1]['delta'])
                    
                    
        #adding weights and bias to their updates    
        for layer in range(len(self.layers)-1):
            if self.layers[layer]["layer_type"] == "input":
                self.layers[layer]['bias'] += self.layers[layer]['bias_update']
            elif self.layers[layer]['layer_type'] == "output":
                self.layers[layer]['weight_matrix'] +=self.layers[layer]['weight_update']
            else:
                self.layers[layer]['weight_matrix'] +=self.layers[layer]['weight_update']
                self.layers[layer]['bias'] += self.layers[layer]['bias_update']
    
    #################################################################

    def calculate_activations(self, prev_layer, current_layer):
        product = np.matmul(prev_layer['activations'], current_layer['weight_matrix']) + prev_layer['bias']
        
        if current_layer['activation_func'] == "sigmoid":
            new_activations = self.sigmoid_activation_func(product)
        elif current_layer['activation_func'] == "relu":
            new_activations = self.relu_activation_func(product)
        elif current_layer['activation_func'] == "softmax":
            new_activations = self.softmax(product)
        else:
            print("sigmoid or relu, you choose")
        
        return new_activations, product
    
    def relu_activation_func(self, x):
        return np.maximum(0, x)
        
    def sigmoid_activation_func(self, x):
        sig = 1/(1 + np.exp(-x))
        return sig    
    
    def softmax(self, x):
        new_array = []
        for datapoint in x:
            stable_datapoint = datapoint - np.max(x)
            new_array.append(np.exp(stable_datapoint)/np.sum(np.exp((stable_datapoint))))
        return np.array(new_array)

    
    #################################################################    

    #Adam
    def Adam(self, current_layer, previous_layer, rho1 = 0.9, rho2 = 0.999, stab = 10e-8):
        gradient = self.SGD(current_layer, previous_layer)
        #I don't know why the t is here or what it does
        
        current_layer['s'] = rho1 * current_layer['s'] + (1 - rho1) * gradient
        current_layer['r'] = rho2 * current_layer['r'] + (1 - rho2) * gradient * gradient #operator between grad is hadamar/elementwise operator

        s_hat = current_layer['s'] / (1 - rho1)
        r_hat = current_layer['r'] / (1 - rho2)
        
        update = s_hat / (r_hat ** 0.5 + stab)
        
        return update
    #Stochastic Gradient Descent
    def SGD(self, current_layer, prev_layer):
        weights = []
        for i in range(len(current_layer['delta'])):
            weights.append(current_layer['delta'][i] * prev_layer['activations'][i][:,np.newaxis])
        return sum(weights) / len(weights)
    
    #backpropogates the cost to the layer-1
    def cost_prime_hidden(self, current_layer,layer_after):
        return np.array(self.activation_prime(current_layer)* np.matmul(layer_after['delta'], layer_after['weight_matrix'].transpose()))

    def weight_update(self, err, activation):
        weights = []
        for i in range(len(activation)):
            weights.append(err[i] * activation[i][:,np.newaxis])
        return weights
    
    def activation_prime(self, layer):
        if layer['activation_func'] == "sigmoid":
            return self.sigmoid_prime(layer['g_inputs'])
        elif layer['activation_func'] == 'relu':
            return self.relu_prime(layer['g_inputs'])

    def sigmoid_prime(self, g_input):
        return self.sigmoid_activation_func(g_input)*(1-self.sigmoid_activation_func(g_input)) 
    
    def relu_prime(self, g_input):
        return np.where(g_input>0, 1.0, 0.0)  
    
    ####################################################################
    #Check that these work for correct size dims
    #Mean Squared Error
    def MSE(self, output_layer, y_batch):
        why = output_layer['activations'] - y_batch
        why_squared = why*why
        MSE = (1/len(y_batch))*np.sum(why_squared)
        return MSE
    
    #Cross Entropy
    def xent(self, output_layer, y_batch):
        return - sum(y_batch * np.log(output_layer))/len(y_batch)
    
    def accuracy_score(self,X, y, cm = False):
        self.forward_pass(X)
        y_predicted = self.layers[-1]['activations']
        accuracy_sum = 0
        y_pred_arg = []
        y_arg = []
        for datapoint in range(len(y)):
            if np.argmax(y[datapoint]) == np.argmax(y_predicted[datapoint]):
                
                
                accuracy_sum += 1
            y_pred_arg.append(np.argmax(y_predicted[datapoint]))
            y_arg.append(np.argmax(y[datapoint]))
        
        if cm == True:
        
            mat = confusion_matrix(y_pred_arg, y_arg)
            sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                        xticklabels=range(10),
                        yticklabels=range(10))
            plt.xlabel('true label')
            plt.ylabel('predicted label')
            plt.show()

        accuracy = accuracy_sum/len(y)
        return accuracy*100
