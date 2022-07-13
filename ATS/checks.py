import pandas as pd

def check_inputs(data=None, cols=None, file_name=None, type_int=None, type_string=None):
    """FOR FUNCTIONS, checks to see if inputs from user are correct. Put all inputs in a list"""
    
    inputVal = True
    
    #Check data
    if data is not None:
        if data[0] is not None:
            for item_data in data:
                if type(item_data) is not pd.core.frame.DataFrame:
                    print("Input Error: 'Data' is not type Pandas DataFrame")
                    inputVal = False
    
    #Check cols
    if cols is not None: #if input is used
        if cols[0] is not None: #if empty list input
            for item_cols in cols:
                if (type(item_cols) is not list):
                    print("Input Error: columns is not a list of strings")
                    inputVal = False
                    
                if (type(item_cols) is list):
                    for item in item_cols:
                        if type(item) is not str:
                            print("Input Error: columns is a list but doesn't contain all strings")
                            inputVal = False
                            break
    
    #Check file name
    if file_name is not None:
        if file_name[0] is not None:
            for item_file in file_name:
                if type(item_file) is not str:
                    print("Input Error: 'file_name' is not a string")
                    inputVal = False

    #Check type integer
    if type_int is not None:
        if type_int[0] is not None:
            for item_int in type_int:
                if type(item_int) is not int:
                    print("Input Error: "+str(item_int)+" is not an integer")
                    inputVal = False

    #Check type string
    if type_string is not None: 
        if type_string is not None:
            for item_str in type_string:
                if type(item_str) is not str:
                    print("Input Error: "+str(item_str)+" is not a string")
                    inputVal = False
        
        
    return inputVal

