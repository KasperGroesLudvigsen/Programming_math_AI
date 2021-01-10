# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 20:40:42 2020

@author: groes
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import statistics
#import math

###### SPLITTING DATASET INTO TRAINING AND VALIDATION #######
# Create an array that for each home makes a boolean. The number of trues are percentage_trainingset %
def create_training_validation_set(df, percentage_trainingset):    
    mask = np.random.rand(len(df)) < percentage_trainingset

    trainDF = pd.DataFrame(df[mask])
    validationDF = pd.DataFrame(df[~mask])

    print(f"Training DF: {len(trainDF)}")
    print(f"Validation DF: {len(validationDF)}")
    
    return trainDF, validationDF 


######## K-FOLD CROSS VALIDATION ########
# Function takes a df and and k (integer) as inputs and returns k training and validation sets
def create_kfold_sets(df, k):
    # Using k fold validation because the dataset is small
    # Shuffling to get representative samples
    df = df.reindex(np.random.permutation(df.index))
    kf = KFold(k)
    fold = 1
    dict_of_sets = {}
    for train_index, validate_index in kf.split(df):
        trainDF = pd.DataFrame(df.iloc[train_index, :])
        validateDF = pd.DataFrame(df.iloc[validate_index])
        dict_name_train = "Fold" + str(fold) + "_training"
        dict_name_validate = "Fold" + str(fold) + "_validation"
        dict_of_sets[dict_name_train] = trainDF
        dict_of_sets[dict_name_validate] = validateDF 
        fold += 1
    return dict_of_sets


# This function creates a variable "balcony"
# The function adds a column of which each element is an integer:
# 0 = no balcony, 1 = possibility of building balcony, 2 = balcony
# We'll use this as in ordered variable, not categorical, as balcony is better,
# therefore larger, than no balcony
def add_balcony_variable(df, no_balcony_value, balcony_possibility_value, balcony_value):
    #home_type_column_index = df.columns.get_loc("Home_type")
    #description_column_index = df.columns.get_loc("description_of_home")

    list_does_home_have_balcony = []
    num_homes_with_no_description = 0

    for description in df["description_of_home"]: #df.iloc[:, description_column_index]:
        if type(description) != str:
            list_does_home_have_balcony.append(0)
            num_homes_with_no_description += 1
            continue
        
        # If the home does not have a balcony but there is an option of adding
        # one, the realtor will typically write "mulighed for altan" or 
        # "altan projekt" (balcony project)
        if "mulighed for altan" in description or "altanprojekt" in description or "altan projekt" in description:
            list_does_home_have_balcony.append(1)
            continue
        if "altan" in description:
            list_does_home_have_balcony.append(2)
            continue
        
        list_does_home_have_balcony.append(0)
        
    df["balcony"] = list_does_home_have_balcony
    print("{} homes had no description".format(num_homes_with_no_description))
    return df


def test_create_balcony_variable():
    test_df = pd.DataFrame()
    test_hometype = ["rækkehus", "villa", "ejerlejlighed", "ejerlejlighed", "ejerlejlighed"]
    test_description = ["lorem ipsum", "lorem ipsum", "this flat has an altan", "this flat has mulighed for altan", "this does not have a balcony"]
    test_df["home_type"] = test_hometype
    test_df["description_of_home"] = test_description
    #test_balcony_variable = create_balcony_variable(test_df, 0, 1)
    #print(test_balcony_variable)
    test_df = add_balcony_variable(test_df, 0, 1, 2)
    assert test_df["balcony"][0] == 0 #[0, 0, 2, 1, 0]    
    assert test_df["balcony"][1] == 0
    assert test_df["balcony"][2] == 2
    assert test_df["balcony"][3] == 1
    assert test_df["balcony"][4] == 0
test_create_balcony_variable()


def make_floor_int(df):
    floor_as_int = []
    for i in range(len(df)):
        # Only house types "villalejlighed" (flat in villa) and "ejerlejlighed" (flat)
        # has floor numbers 
        if df["Home_type"][i] == "Villalejlighed" or df["Home_type"][i] == "Ejerlejlighed":
            try:
                floor_as_int.append(int(df["floor"][i][0]))
            except:
                median_value = int(round(statistics.median(floor_as_int)))
                floor_as_int.append(median_value)
                print("Error converting floor to int in line {}. Inserting median value: {}".format(i, median_value))
        else:
            floor_as_int.append(0)
    df["floor_as_int"] = floor_as_int
    return df #floor_as_int




def get_zips_to_be_grouped(df, threshold):
    """
    Helper function for add_zip_code_variable()
    
    If an area has more than one zipcode (e.g. Frederiksberg C), those of 
    the zipcodes that account for less than 1 % (per default) of datapoints,
    all zip codes within the area will be grouped into 1 zipcode

    Parameters
    ----------
    df : Pandas dataframe
        Df with data
        
    threshold : Float
        If a zip code accounts for fewer than 'threshold' datapoints, it will
        be grouped into one

    Returns
    -------
    zips_to_be_grouped : SET

    """
    zip_code_occurences = df.zip_code_town.value_counts()
    zips_to_be_grouped = []

    threshold = len(df) * threshold
    print("Grouping zip codes with fewer than {} datapoints".format(threshold))
    for i in range(len(zip_code_occurences)):
        area = zip_code_occurences.index[i]

        if zip_code_occurences[i] < threshold:
            zips_to_be_grouped.append(area)
    
    # using set() for higher look up speed
    return set(zips_to_be_grouped)



def add_zip_code_variable(df, threshold=0.01):
    """
    Some zip codes in Copenhagen cover very small areas whereas others cover
    very large areas. The zip codes covering small areas are not well repre-
    sented in the dataset. Therefore, we group zip codes that have few datapoints
    in groups that represent the area of Copenhagen the zip code belongs to. 
    E.g. 

    Parameters
    ----------
    df : PANDAS DATAFRAME
        df with data
        
    threshold : FLOAT
        If a zip code accounts for fewer than 'threshold' datapoints, it will
        be grouped into one

    Returns
    -------
    Enriched df

    """
    
    zips_to_be_grouped = get_zips_to_be_grouped(df, threshold)
    zipcodes = []
    
    for i in range(len(df)):
        area = df.zip_code_town[i]
        if area in zips_to_be_grouped:
            if "København V" in area:
                zipcodes.append("1600")
            if "København K" in area:
                zipcodes.append("1300")
            if "Frederiksberg C" in area:
                zipcodes.append("1900")
        else:
            # The first element of the string 'area' is supposed to be the zipcode
            zipcode = area[:4]
            try:
                int(zipcode)
            except:
                print("{} in row {} of zip_code_town is not a number".format(zipcode, i))
                zipcodes.append("NaN")
            else:
                zipcodes.append(zipcode)
    
    df["zipcodes"] = zipcodes
    
    return df



def make_m2_price(df):
    df["m2_price"] = df.list_price_dkk / df.home_size_m2
    return df


    
def enrich_dataset(df, no_balcony_value, balcony_possibility_value, balcony_value):
    """
    Wrapper for the methods that engineers new features

    """
    df = add_balcony_variable(df, no_balcony_value, balcony_possibility_value, balcony_value)
    df = make_floor_int(df)
    df = add_zip_code_variable(df)
    df = make_m2_price(df)
    return df
    
def add_neighboorhood_avg_m2_price(df):
    grouped_df = df.groupby("zipcodes")
    mean_df = grouped_df.mean()
    mean_df = mean_df.reset_index()
    zipcode_avg_m2_price = []
    for i in range(len(df)):
        #print(i)
        zipcode = df["zipcodes"][i]
        #print("zipcode")
        #print(zipcode)
        index = mean_df.index[mean_df["zipcodes"] == zipcode] #mean_df["zipcodes"].index(zipcode)
        avg_price = float(mean_df.iloc[index, -1])
        #print(avg_price)
        zipcode_avg_m2_price.append(avg_price)
    df["zipcode_avg_m2_price"] = zipcode_avg_m2_price
    return df 
    
    

# INVESTIGATING WHICH WORDS ARE USUALLY AROUND "ALTAN" (Danish for "balcony")
def get_words_surrounding_altan(df):
    description_column_index = df.columns.get_loc("description_of_home")
    list_descriptions_splitted = []
    
    for description in df.iloc[:, description_column_index]:
        description = str(description)
        # Making list of descriptions split by space such that indivdual words
        # can be indexed
        list_descriptions_splitted.append(description.split())
    
    list_of_text_surrounding_altan = []
    list_of_mulighed_for_altan = []
    for i in list_descriptions_splitted:
        if "altan" in i:
            position_of_altan = i.index("altan")
            start = position_of_altan - 5
            if position_of_altan == 0:
                start = 0
            end = position_of_altan + 6
            text_surrounding_altan = i[start:end]
            list_of_text_surrounding_altan.append(text_surrounding_altan)
            if i[position_of_altan-2] == "mulighed":
                list_of_mulighed_for_altan.append(text_surrounding_altan)
    
    return list_of_text_surrounding_altan, list_of_mulighed_for_altan



