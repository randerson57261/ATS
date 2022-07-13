import pandas as pd
import arcgis
from arcgis.gis import GIS
import xlsxwriter
from ATS.checks import check_inputs
from datetime import datetime, timedelta


def arcGIS_data_load_alt(arcGIS_ID, mute=False):
    """Loads data from arcGIS online   
    INPUT: arcGIS_ID = string, arcGIS feature layer ID. On arcGIS online, navigate to feature's page. The ID is the unique code at the end of the web page's address.
        mute = True/False. Default is False, turns off display of feature service link
    RETURNS: dictionary of outputs. Keys = 'data0', 'data1', 'data2' etc. for layers in feature. Also returns 'feature_service' entry
    To get 1st layer from feature service add line of code below this function:
    data = output['data0']
    """

    # connect to arcGIS online
    gis = GIS('https://a-t-s.maps.arcgis.com',
              '###########', '###############')

    # get by direct ID
    feature_service = gis.content.get(arcGIS_ID)

    if not mute:
        display(feature_service)

    # Create and add entires to output dictionary
    output = {}  # Create empty dictionary
    if feature_service.layers:  # if there are feature layers
        for i, layer in enumerate(feature_service.layers):
            # If there are features in feature layer
            if layer.query(return_count_only=True) > 0:
                keyName = 'data'+str(i)  # Key:
                # Update dictionary
                output.update(
                    {keyName: pd.DataFrame.spatial.from_layer(layer)})

    if feature_service.tables:  # If there are tables
        for i, table in enumerate(feature_service.tables):
            if table.query(return_count_only=True) > 0:  # If there are records in table
                keyName = 'table'+str(i)  # Key
                # update dictionary
                output.update(
                    {keyName: pd.DataFrame.spatial.from_layer(table)})

    # Add feature service entry to dictionary
    output.update({'feature_service': feature_service})

    return output


def arcGIS_data_load(arcGIS_ID, mute=False):
    """Loads data from arcGIS online   
    INPUT: arcGIS_ID = string, arcGIS feature layer ID. On arcGIS online, navigate to feature's page. The ID is the unique code at the end of the web page's address.
        mute = True/False. Default is False, turns off display of feature service link
    RETURNS: dictionary of outputs. Keys = 'data0', 'data1', 'data2' etc. for layers in feature. Also returns 'feature_service' entry
    To get 1st layer from feature service add line of code below this function:
    data = output['data0']
    """

    # connect to arcGIS online
    gis = GIS('https://a-t-s.maps.arcgis.com',
              '############', '##############')

    # get by direct ID
    feature_service = gis.content.get(arcGIS_ID)

    if not mute:
        display(feature_service)

    # Create and add entires to output dictionary
    output = {}  # Create empty dictionary
    if feature_service.layers:  # if there are feature layers
        for i, layer in enumerate(feature_service.layers):
            # If there are features in feature layer
            if layer.query(return_count_only=True) > 0:
                keyName = 'data'+str(i)  # Key:
                # Update dictionary
                output.update({keyName: layer.query().sdf})

    if feature_service.tables:  # If there are tables
        for i, table in enumerate(feature_service.tables):
            if table.query(return_count_only=True) > 0:  # If there are records in table
                keyName = 'table'+str(i)  # Key
                # update dictionary
                output.update({keyName: table.query().sdf})

    # Add feature service entry to dictionary
    output.update({'feature_service': feature_service})

    return output


def year_subset(data, sYear, mute=None):
    """Subsets data for desired year
    INPUTS: data = pandas dataframe from arcGIS import
                Date column needs to be named "Observation_Date"
            sYear = string, starting year to subset data by.
            mute = True, if you do not want text printed to consol
    RETURNS: subsetted pandas dataframe"""

    sDate = sYear+"-10-01 07:00"
    eDate = str(int(sYear)+1)+"-10-01 07:00"
    sub = data.loc[(data["Observation_Date"] >= sDate) &
                   (data["Observation_Date"] < eDate), :]

    if mute == None:
        print("Analysis for", sDate, "UTC to", eDate, "UTC")

    if len(sub.index) > 0:
        if mute == None:
            print("Entries in Subset:", len(sub.index))
        return sub
    else:
        if mute == None:
            print("ERROR: No Data in Subset")
        return sub


def combine_RT_data(records, layer, ref_name='Feature_ID'):
    """Combines layer and it's record table data
    INPUT: 
        Records = pandas dataframe, data from related table
        Layer = pandas dataframe, data from feature layer
    OUTPUT: merged pandas dataframe"""
    records[ref_name] = records[ref_name].str.strip('{}').str.lower()
    merged = records.merge(layer, left_on=ref_name,
                           right_on='GlobalID', how='inner')
    merged = merged.drop(columns=[ref_name, 'GlobalID_y', 'OBJECTID_y'])
    merged = merged.rename(
        columns={'OBJECTID_x': 'OBJECTID', 'GlobalID_x': 'GlobalID'})

    return merged


def get_constants(selection):
    """Gets commonly used constants
    INPUT: selection = string of constant's name (options: 'tree_survival_plot_areas', 'habitat_area', 'trees_planted_count')
    RETURNS: dictionary of keys:values"""

    if selection == 'tree_survival_plot_areas':
        # Area in m2 of tree sampling plots
        return pd.Series({'Oak Riparian': 100, 'Oak Savannah': 400, 'Oak Woodland': 100, 'Sycamore Riparian': 100})
    elif selection == 'habitat_area':
        # Acres
        return pd.Series({'Oak Riparian': 7.0458978653000, 'Oak Savannah': 81.7684504996000, 'Oak Woodland': 60.6254320951000, 'Sycamore Riparian': 28.2267564168000})
    elif selection == 'trees_planted_count':
        return pd.Series({'Oak Riparian': 1776, 'Oak Savannah': 4117, 'Oak Woodland': 14029, 'Sycamore Riparian': 4727})
    elif selection == 'initial_planting_density':
        return pd.Series({'Oak Riparian': 252, 'Sycamore Riparian': 167})


def export_data_deliverable(data, feature, field_order, layer_num=0, table_num=0, file_name=None):
    """Output excel document of formated data deliverable
    INPUTS: data = pandas dataframe of all data from feature
    feature = feature layer, key= "feature_layer" from output of arcGIS_data_load()
    field_order = list of strings, column order
    layer_num = (optional, defualt = 0) integer, layer number of feature service
    table_num = (optional, defualt = 0) integer, table number of feature service
    file_name = (optional), name of file. Include up to 'Data'. Function adds '_Data_', current date, and '.xlsx'

    RETURNS: none, outputs file to G:\Scripts_Toolboxes\Annual_Report\Exports
    """
    # Check inputs
    if check_inputs(data=[data], file_name=[file_name]):

        # Sort data
        data = data.sort_values(by='Observation_Date', ascending=False)

        # check if input has the right number of columns
        if len(field_order) == len(data.columns):
            # Reorder fields
            data = data[field_order]

            # Feature - Create Field Alias dictionary
            fields = feature.layers[layer_num].properties.fields

            fn = []
            fa = []
            for field in fields:
                fn.append(field['name'])
                fa.append(field['alias'])

            F_alias_dictionary = dict(zip(fn, fa))
            # Rename Columns
            renamed_data = data.rename(columns=F_alias_dictionary)

            # Table - Create Table Alias Dictionary
            try:
                tFields = feature.tables[table_num].properties.fields

                tn = []
                ta = []
                for tField in tFields:
                    tn.append(tField['name'])
                    ta.append(tField['alias'])

                T_alias_dictionary = dict(zip(tn, ta))
                renamed_data.rename(columns=T_alias_dictionary, inplace=True)

            except:
                print('No Related Table Found')

            # Rename date column
            renamed_data.rename(
                columns={"Observation Date": "Observation Date (UTC)"}, inplace=True)
            renamed_data.rename(
                columns={"Observation_Date": "Observation Date (UTC)"}, inplace=True)

            # File name
            crtDate = datetime.today().strftime('%Y%m%d')
            if not file_name:
                filename = 'Outputs/' + \
                    feature.layers[layer_num].properties.name + \
                    '_Data_'+crtDate+'.xlsx'
            else:
                filename = 'Outputs/'+file_name+'_Data_'+crtDate+'.xlsx'

            # Open excel writing instance
            writer_object = pd.ExcelWriter(filename,
                                           engine='xlsxwriter',
                                           datetime_format='yyyy-mm-dd')

            # Write a dataframe to the worksheet.
            renamed_data.to_excel(
                writer_object, sheet_name='Data', index=False)

            # Create xlsxwriter worksheet object
            workbook = writer_object.book

            worksheet = writer_object.sheets['Data']

            # set width of the B and C column
            data_format = workbook.add_format(
                {'align': 'center', 'valign': 'vcenter'})

            worksheet.set_column('A:A', 10, data_format)
            worksheet.set_column('B:D', 20, data_format)
            worksheet.set_column('D:BZ', 20, data_format)
            worksheet.set_row(0, 25)

            # object and output the Excel file.
            writer_object.save()

            print('Exported to '+filename)

        else:
            print('Incorrect number of columns in field_order list')


def obs_date_to_water_year(obs_date):
    """Converts observation date to water year. 
    INPUT
    obs_date = Pandas Series of datetime entries. Actual observation date
    returns Pandas series of strings with water year"""

    offset = timedelta(days=274, hours=7)
    date_offset = obs_date - offset

    s_year = date_offset.dt.year.apply(str)
    e_year = '-' + ((date_offset.dt.year + 1).apply(str))

    water_year = s_year.str.cat(e_year)
    return water_year
