from ATS.checks import check_inputs
from ATS.general import arcGIS_data_load
from ATS.QAQC import text_cleaner

import pandas as pd

def update_Cal_IPC(item_id, IPC_col_name, species_col_name, layer=None, table=None):
    """Updates Cal IPC Rating column in feature layer or table
    INPUT: item_id = string, ArcGIS item ID
    IPC_col_name = string, name of IPC Rating column in dataset
    species_col_name = string, name of species column in dataset

    ONLY input one of the following:
    layer = integer, layer to update
    table = integer, table to update

    RETURNS: None, displays messages"""

    
    #Check inputs are valid
    valid = False

    if (layer is not None) & (table is not None):
        print('Input Error: Only input a layer or table, not both')
            
    elif layer is not None:
        valid = check_inputs(type_string=[item_id,IPC_col_name, species_col_name],type_int=[layer])

            
    elif table is not None:
        valid = check_inputs(type_string=[item_id,IPC_col_name, species_col_name],type_int=[table])
        print('here')
            
    else:
        print('Input Error: No layer or table input')


    #If inputs are valid...
    if valid:
        output = arcGIS_data_load(item_id)
        feature_service = output['feature_service']

        
        #Get Species list
        pltDir = r"/home/user/SAC/Script Support Files/SAC Master Plant Species List 20191114.xlsx"
        pltSheet = "Year 5 Plant Species List"

        print("Current Plant Species List file location:\n"+pltDir+"\nCurrent Sheet Name:\n"+pltSheet)

        # Will need to update this file locally on the G drive and change the sheet name
        pltLst = pd.read_excel(pltDir, pltSheet)

        #Clean lists
        for col in ['Cal-IPC Rating','Species']:
            pltLst[col] = text_cleaner(pltLst[col])
                
        #Get CAL IPC High list
        Cal_IPC_H = pltLst.loc[pltLst['Cal-IPC Rating'] == "High", 'Species'].tolist()
                
        #Get CAL IPC Moderate list
        Cal_IPC_M = pltLst.loc[pltLst['Cal-IPC Rating'] == "Moderate", 'Species'].tolist()
            
        #Update records
        # Get all records
        if layer is not None:
            VC_records = feature_service.layers[layer].query(return_all_records=True)
        else:
            VC_records = feature_service.tables[table].query(return_all_records=True)
            
        edits = []

        #loop through each feature
        for feature in VC_records.features:
            
            species = feature.attributes[species_col_name]
            update_flag = False

            if species in Cal_IPC_H:
                
                feature.attributes[IPC_col_name] = 'High'
                update_flag = True

            elif species in Cal_IPC_M:
                feature.attributes[IPC_col_name] = 'Moderate'
                update_flag = True

            else:
                feature.attributes[IPC_col_name] = 'None'
                update_flag = True
                
            #if an change was made, update through API
            if update_flag:
                edits.append(feature)
                
        #send updates if edit was made
        if edits:
            if layer is not None:
                result = feature_service.layers[layer].edit_features(updates=edits)
            else:
                result = feature_service.tables[table].edit_features(updates=edits)

            #Check if update was successful
            fail_flag = False
            for item in result['updateResults']:
                if not item['success']:
                    fail_flag = True
                    print('Could not update feature, Object ID: ',item['objectId'])

            if not fail_flag:
                print('Successfully updated features')
                
        else:
            print('No edits made')
                
            
