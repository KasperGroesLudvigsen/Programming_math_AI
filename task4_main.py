# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 21:22:59 2020

@author: groes
"""
#Test comment
import task4_data as t4
#import neural_network as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import copy
import torch
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
import torchvision   
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import math 
import pytorch_dataset as MPD # class for creating pytorch datasets for task 4


##############################################################################
####---------- FEATURE ENGINEERING AND DATA PRE PROCESSING ---------------####
##############################################################################
def data_preprocessing(exclude_cooperative_dwellings, columns_to_use, test_size,
                       batchsize, epochs, learningrate, feature_scaling,
                       no_balcony_value, balcony_possibility_value, balcony_value,
                       experiment):

    model_settings = {
        "experiment" : experiment,
        "test_size" : [test_size],
        "batchsize_nn" : [batchsize],
        "epochs_nn" : [epochs],
        "learning_rate" : [learningrate],
        "feature_scaling" : [feature_scaling],
        "balcony_value" : [balcony_value],
        "no_balcony_value" : [no_balcony_value],
        "balcony_possibility_value" : [balcony_possibility_value]
        }
    
      
    df = pd.read_csv("df_all_data_w_desc_2020-12-14.csv")
    
    # Removing cooperative dwellings
    if exclude_cooperative_dwellings:
        num_datapoints = len(df)
        index = df[df["Home_type"] == "Andelsbolig"].index
        df.drop(index, inplace=True)
        print("{} cooperative dwellings removed".format(num_datapoints-len(df)))
        df = df.reset_index(drop=True)
        
    df = t4.enrich_dataset(df, no_balcony_value, balcony_possibility_value, balcony_value)
    
    # Removing duplicates
    num_datapoints = len(df)
    df = df.drop_duplicates(subset="home_url_realtor", keep="first")
    print("{} duplicates removed".format(num_datapoints-len(df)))
    df = df.reset_index(drop=True)
    
    # Adding age feature and removing "built_year"
    df["age"] = 2020 - df["built_year"]
    df.drop(["built_year"], axis=1, inplace=True)
    columns_to_use.append("age")
    
    # Adding variable representing the average m2 price in the neighborhood
    df = t4.add_neighboorhood_avg_m2_price(df)
    
    # Removing variables that we wont be using
    columns = df.columns
    columns_to_drop = list(set(columns) - set(columns_to_use))
    df = df.drop(columns_to_drop, axis=1)
    
    # Identifying rows with nans
    is_NaN = df.isnull()
    row_has_NaN = is_NaN.any(axis=1)
    rows_with_NaN = df[row_has_NaN]
    
    # Replacing nans
    for index in df.index:
        if math.isnan(df.loc[index, "rooms"]):
            if df.loc[index, "home_size_m2"] < 40:
                df.loc[index, "rooms"] = 1 #df.at[home_size_m2_idx, i] = 1
            elif df.loc[index, "home_size_m2"] < 70:
                df.loc[index, "rooms"] = 2
            elif df.loc[index, "home_size_m2"] < 100:
                df.loc[index, "rooms"] = 3
            else: 
                df.loc[index, "rooms"] = 4
        if math.isnan(df.loc[index, "lotsize_m2"]):
            if df.loc[index, "Home_type"] == "Ejerlejlighed" or df.loc[index, "Home_type"] == "Andelsbolig":
                df.loc[index, "lotsize_m2"] = 0
            else:
                df.loc[index, "lotsize_m2"] = df["lotsize_m2"].mean()
        if math.isnan(df.loc[index, "expenses_dkk"]):
            df.loc[index, "expenses_dkk"] = df["expenses_dkk"].mean()
        if math.isnan(df.loc[index, "age"]):
            df.loc[index, "age"] = round(df["age"].mean())
    
    # Feature scaling
    features_to_scale = list(df.columns)
    features_to_scale.remove("Home_type")
    features_to_scale.remove("zipcodes")
    
    df_for_scaling = copy.copy(df)
    df_for_scaling.drop(["Home_type", "zipcodes"], axis=1, inplace=True)
    
    if feature_scaling == "standardise":
        for feature in features_to_scale:
            scaler = StandardScaler().fit(df[[feature]]) #.reshape(-1, 1))
            df_for_scaling[feature] = scaler.transform(df_for_scaling[[feature]])
        df.drop(features_to_scale, axis=1, inplace=True)
        df = pd.concat([df, df_for_scaling], axis=1)
        
    if feature_scaling == "normalise":
        norm = MinMaxScaler().fit(df_for_scaling)
        normalized_data = norm.transform(df_for_scaling)
        df_normalized = pd.DataFrame(data=normalized_data, columns=features_to_scale)
        df.drop(features_to_scale, axis=1, inplace=True)
        df = pd.concat([df, df_normalized], axis=1)
        
    # One hot encoding zipcodes
    onehot_encoded_variables = []
    zipcodes_onehot = pd.get_dummies(df.zipcodes, prefix="Zipcode")
    df = pd.concat([df, zipcodes_onehot], axis=1)
    onehot_encoded_variables.append("zipcodes")
    
    # One hot encoding home_type
    hometype_onehot = pd.get_dummies(df.Home_type, prefix="Hometype")
    df = pd.concat([df, hometype_onehot], axis=1)
    onehot_encoded_variables.append("Home_type")
    
    # Removing variables that have been onehot encoded
    df = df.drop(onehot_encoded_variables, axis=1)
    
    # Splitting data
    X = df.drop(["m2_price"], axis=1)
    y = df["m2_price"]
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=101)
    
    return X_train, X_test, y_train, y_test, model_settings, df



##############################################################################
####-------------- IMPLEMENTING NEURAL NETWORK MODEL ---------------------####
############################################################################## 

def run_nn(epochs, X_train, X_test, y_train, y_test, model_settings):
    class FFN(nn.Module):
        def __init__(self):
            super(FFN, self).__init__()
            self.linear1 = nn.Linear(X_train.shape[1], 50) 
            self.linear2 = nn.Linear(50, 50)
            self.linear3 = nn.Linear(50, 50)
            self.linear4 = nn.Linear(50, 50)
            self.linear5 = nn.Linear(50, 50)
            self.linear6 = nn.Linear(50, 40)
            self.final_linear = nn.Linear(40, 1) 
            self.activation = nn.Sigmoid() #nn.Sigmoid() #nn.ReLU()
            
        def forward(self, inputs):
            #x = inputs.view(-1, 28*28)
            x = inputs
            x = self.activation(self.linear1(x))
            x = self.activation(self.linear2(x))
            x = self.activation(self.linear3(x))
            x = self.activation(self.linear4(x))
            x = self.activation(self.linear5(x))
            x = self.activation(self.linear6(x))
            #x = self.activation(self.linear7(x))
            #x = self.activation(self.linear8(x))
            #x = self.activation(self.linear9(x))
            #x = self.activation(self.linear10(x))
            #x = self.activation(self.linear11(x))
            x = self.final_linear(x)
            return x
    
    
    training_dataset = MPD.MakePytorchData(X_train, y_train)
    train_loader = DataLoader(dataset=training_dataset, batch_size=batchsize, shuffle=True)
    
    model = FFN()
    model_settings["model_spec_nn"] = model
    nn_criterion = nn.MSELoss()
    model_settings["nn_criterion"] = nn_criterion
    optimizer_nn = torch.optim.SGD(params=model.parameters(), lr=learningrate)
    model_settings["nn_optimizer"] = optimizer_nn
    
    for epoch in range(epochs):

        for i, (inputs, labels) in enumerate(train_loader):
            
            # Changing dimensions of output so it matches labels
            outputs_nn = model(inputs)
            outputs_nn_shape = outputs_nn.shape[0]
            outputs_nn = outputs_nn.view(outputs_nn_shape)
        
            model.zero_grad()
            loss_nn = nn_criterion(outputs_nn, labels)
            loss_nn.backward() 
            
            optimizer_nn.step()
            
    test_dataset = MPD.MakePytorchData(X_test, y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batchsize, shuffle=True)
       
    model.eval()
    loss_nn = 0
    iterations = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            iterations += 1
            outputs = model.forward(inputs)
            y_pred = outputs
            y_pred_shape = y_pred.shape[0]
            y_pred = y_pred.view(y_pred_shape)
            loss = nn_criterion(y_pred, labels)
            loss_nn += loss.item()
        final_loss_nn = loss_nn / iterations
        print("Final loss is: {} ".format(final_loss_nn))
        model_settings["loss_nn"] = final_loss_nn
      
    return model_settings
    

##############################################################################
####------------- IMPLEMENTING LINEAR REGRESSION MODEL -------------------####
##############################################################################

def run_lin_reg(lin_reg_epochs, X_train, X_test, y_train, y_test, model_settings):
    #### Converting datasets to pytorch tensors
    model_settings["lin_reg_epochs"] = lin_reg_epochs
    
    # Defining model
    linear_regression_model = nn.Linear(in_features=X_train.shape[1], out_features=1)
    criterion_lin_reg = nn.MSELoss()
    model_settings["criterion_lin_reg"] = criterion_lin_reg
    optimizer_lin_reg = torch.optim.SGD(params=linear_regression_model.parameters(), lr=learningrate)
    model_settings["lin_reg_optimizer"] = optimizer_lin_reg
    
    # Preparing data

    X_lin_reg = torch.from_numpy(X_train.values).float()

    y_lin_reg = torch.from_numpy(y_train.values).float()
    y_lin_reg = y_lin_reg.view(X_train.shape[0], 1)

    # Training
    for epoch in range(lin_reg_epochs):
        y_pred = linear_regression_model(X_lin_reg)
        loss = criterion_lin_reg(y_pred, y_lin_reg)
        loss.backward()
        optimizer_lin_reg.step()
        optimizer_lin_reg.zero_grad()

    test_dataset = MPD.MakePytorchData(X_test, y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batchsize, shuffle=True)
    
    # Testing 
    print("TESTING:")
    linear_regression_model.eval()
    num_datapoints = 0
    loss_lin_reg = 0
    iterations = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            iterations += 1
            num_datapoints += len(inputs)
            outputs_lin_reg = linear_regression_model(inputs)
            outputs_lin_reg = outputs_lin_reg.view(outputs_lin_reg.shape[0])
            loss = criterion_lin_reg(outputs_lin_reg, labels)
            loss_lin_reg += loss.item()
        final_loss_lin_reg = loss_lin_reg / iterations
        print("Final loss is: {} ".format(final_loss_lin_reg))
        model_settings["loss_lin_reg"] = final_loss_lin_reg
    
    return model_settings



####------------------ EXPERIMENTING WITH DIFFERENT SETTINGS -------------####

# Settings
exclude_cooperative_dwellings = True
columns_to_use = ["Home_type", "rooms", "home_size_m2", "built_year",
                  "lotsize_m2", "expenses_dkk", "floor_as_int", "balcony",
                  "zipcodes", "m2_price"]
test_size = 0.3
batchsize = 40
epochs_nn = 1
learningrate = 0.001
feature_scaling = "normalise" # should either be "standardise" or "normalise"

no_balcony_value = 0 # to be used in creating a variable reflecting whether the home has a balcony
balcony_possibility_value = 2 # to be used in creating a variable reflecting whether the home has a balcony
balcony_value = 5 # to be used in creating a variable reflecting whether the home has a balcony
epochs_lin_reg = epochs_nn

experiment = "increased number of layers in nn and more nodes"

# Generating data
X_train, X_test, y_train, y_test, model_specs, df = data_preprocessing(exclude_cooperative_dwellings,
                                                                   columns_to_use,
                                                                   test_size,
                                                                   batchsize,
                                                                   epochs_nn,
                                                                   learningrate,
                                                                   feature_scaling,
                                                                   no_balcony_value,
                                                                   balcony_possibility_value,
                                                                   balcony_value,
                                                                   experiment)

# Running experiments

lin_reg_model_spec = run_lin_reg(epochs_lin_reg, X_train, X_test, y_train, y_test, model_specs)
nn_model_specs = run_nn(epochs_nn, X_train, X_test, y_train, y_test, lin_reg_model_spec)
print("Linear regression model details:")
print(lin_reg_model_spec)
print("Neural network model details:")
print(nn_model_specs)

"""
lin_reg_df = pd.DataFrame.from_dict(data=lin_reg_model_spec)
nn_df = pd.DataFrame.from_dict(data=nn_model_specs)

all_results = pd.concat([all_results, lin_reg_df, nn_df])

all_results.to_excel("task4_experimentation_results.xlsx")

all_results_no_duplicates = all_results.drop_duplicates(keep="first")

descr_stats_loss_nn = all_results_no_duplicates["loss_nn"].describe()
descr_stats_loss_lin_reg = all_results_no_duplicates["loss_lin_reg"].describe()
descr_stats_all = pd.DataFrame()


### Plotting m2 price against home size
import seaborn as sns
from matplotlib import pyplot as plt

# All data
_, _, _, _, _, df = data_preprocessing(exclude_cooperative_dwellings,
                                        columns_to_use,
                                        test_size,
                                        batchsize,
                                        epochs_nn,
                                        learningrate,
                                        feature_scaling,
                                        no_balcony_value,
                                        balcony_possibility_value,
                                        balcony_value,
                                        experiment)


ax = sns.lineplot(x="home_size_m2", y="m2_price", data=df)


plt.scatter(x="rooms", y="m2_price", data=df)
"""
