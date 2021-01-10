# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 12:45:05 2020

@author: groes
"""
import pandas as pd
import task4_data as t4d

def test_create_balcony_variable():
    test_df = pd.DataFrame()
    test_hometype = ["rækkehus", "villa", "ejerlejlighed", "ejerlejlighed", "ejerlejlighed"]
    test_description = ["lorem ipsum", "lorem ipsum", "this flat has an altan", "this flat has mulighed for altan", "this does not have a balcony"]
    test_df["home_type"] = test_hometype
    test_df["description_of_home"] = test_description
    #test_balcony_variable = create_balcony_variable(test_df, 0, 1)
    #print(test_balcony_variable)
    test_df = t4d.add_balcony_variable(test_df, 0, 1, 2)
    assert test_df["balcony"][0] == 0 #[0, 0, 2, 1, 0]    
    assert test_df["balcony"][1] == 0
    assert test_df["balcony"][2] == 2
    assert test_df["balcony"][3] == 1
    assert test_df["balcony"][4] == 0

test_create_balcony_variable()


def unittest_make_floor_int():
    df = pd.read_csv("df_all_data_w_desc_2020-12-14.csv")
    df = t4d.make_floor_int(df)
    assert "floor_as_int" in df.columns
    
unittest_make_floor_int()


def unittest_add_zip_code_variable():
    df = pd.read_csv("df_all_data_w_desc_2020-12-14.csv")
    df_zips = t4d.add_zip_code_variable(df)
    raised = False
    for zipcode in df_zips["zipcodes"]:
        assert len(zipcode) == 4
        try:
            int(zipcode)
        except:
            raised = True
    assert raised == False
    
unittest_add_zip_code_variable()


def unittest_add_zip_code_variable():
    df = pd.read_csv("df_all_data_w_desc_2020-12-14.csv")
    df_zips = add_zip_code_variable(df)
    raised = False
    for zipcode in df_zips["zipcodes"]:
        assert len(zipcode) == 4
        try:
            int(zipcode)
        except:
            raised = True
    assert raised == False
    
unittest_add_zip_code_variable()

def unittest_make_floor_int():
    df = pd.read_csv("df_all_data_w_desc_2020-12-14.csv")
    df = make_floor_int(df)
    assert "floor_as_int" in df.columns
unittest_make_floor_int()
