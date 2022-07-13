from ATS.checks import check_inputs
from ATS.general import year_subset
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import numpy as np
import pandas as pd


def find_blank_entries(data, exclude_cols=None):
    """Checks data for NA entries
    INPUTS: data = pandas dataframe
            exclude_cols = optional, list of strings of column names to exclude from analysis ex. ['Var_1','Var_2']
    RETURNS: displays empty entries"""
    pd.set_option('display.max_colwidth', 70)
    
    if check_inputs(data=[data], cols=[exclude_cols]):
        
        #Get list of columns that are data type == object (strings)
        tCols = data.columns

        #Remove excluded columns
        if exclude_cols:
            tCols = list(set(tCols) - set(exclude_cols))

        for col in tCols:
            if data.loc[:,col].isna().any() == True:
                print('Blank',col,'entrie(s) exist!')
                display(data.loc[data.loc[:,col].isna(),:])
            else:
                print('No blank',col)

def find_outlier(data,check_cols, boxplot=True):
    """Finds outliers in continuous data by IQR
    INPUTS: data = pandas dataframe
            check_cols = list of strings of column names to analyze ex. ['Var_1','Var_2']
            boxplot = True/False, Displays boxplot if True, optional
            
    RETRUNS: dataframe with new boolean column of outlier detection
        Also displays outlier entries"""
    output = data.copy()
    
    for col in check_cols:
        # QAQC Find outliers by IQR
        Q1 = data.loc[:,col].quantile(0.25)
        Q3 = data.loc[:,col].quantile(0.75)
        IQR = Q3 - Q1
        output[col+'_Outlier'] = (data.loc[:,col] < (Q1 - 1.5 * IQR)) |(data.loc[:,col] > (Q3 + 1.5 * IQR))
        print(col,' Outlier:',output.loc[:,col+'_Outlier'].any())

        #Display outlier
        if output.loc[:,col+'_Outlier'].any():
            print('See Outlier Entries Below:')
            display(output.loc[output.loc[:,col+'_Outlier'],:])
            
            #Create boxplot
            if boxplot:
                fig, ax = plt.subplots(figsize=(10,2))
                ax.boxplot(output.loc[:,col].dropna(), vert=False, widths=[0.75])
                plt.show()
            
    return output

def QAQC_plot(data, cols_to_plot):
    """QAQC - Visualize Data in pie charts or histograms.
    Function detects data type and displays appropriate plot.
    INPUTS: data = pandas dataframe
            cols_to_plot = list of strings of column names to analyze ex. ['Var_1','Var_2']
    RETURNS: displays plots"""
    #GG plot style
    plt.style.use('ggplot')

    
    #Create figure based on number of columns
    rows = math.ceil(len(cols_to_plot)/2)
    fig, axs = plt.subplots(rows, 2, figsize=(14, 7*rows), facecolor='white')
    
    #Loop through each column and plot data
    for i in range(0,len(cols_to_plot)):
        col = cols_to_plot[i]
        
        #Get index of subplot
        if i%2 == 0: #If even
            plt_c = 0      #Plot Column
            plt_r = int(i/2)  #Plot row
        if i%2 == 1: #If odd
            plt_c = 1       #Plot column
            plt_r = int(i/2-.5)   #Plot row
                
        #Test if there are values in column
        if data.loc[:,col].isna().all() == False: 

            dataType = type(data.loc[data[col].notnull(),col].values[0])          
            #If data is string, plot pie chart
            if (dataType==str): 
                count = data.loc[:,col].value_counts(dropna=False)
                colormap=cm.Blues(np.arange(len(count))/len(count))

                if rows == 1: #If one row of plots (axs object has only 1 index when 1 row)
                    axs[plt_c].pie(count, labels=count.index, shadow=True, colors=colormap)
                    axs[plt_c].set_title(col)
                else:        #If more than one row of plots (axs object has 2 indexes when >1 row)
                    axs[plt_r,plt_c].pie(count, labels=count.index, shadow=True, colors=colormap)
                    axs[plt_r,plt_c].set_title(col)

            #If data is float or int, plot histogram
            elif (dataType == np.float64) | (dataType == np.int64):
                if rows == 1:      #If one row of plots (axs object has only 1 index when 1 row)
                    axs[plt_c].hist(data.loc[:,col],color='lightslategrey',edgecolor='black')
                    axs[plt_c].set_xlabel(col)
                    axs[plt_c].set_ylabel('Frequency') 
                else:              #If more than one row of plots (axs object has 2 indexes when >1 row)
                    axs[plt_r,plt_c].hist(data.loc[:,col],color='lightslategrey',edgecolor='black')
                    axs[plt_r,plt_c].set_xlabel(col)
                    axs[plt_r,plt_c].set_ylabel('Frequency') 

            #If Date column
            elif dataType == np.datetime64:
                #Group date by day
                r = data.groupby([data[col].dt.date]).count()

                #Convert datetime index to string
                r['DateString'] = r.index
                r['DateString'] = r['DateString'].apply(lambda x: x.strftime('%Y-%m-%d'))

                if rows == 1:      #If one row of plots (axs object has only 1 index when 1 row)
                    axs[plt_c].bar(r['DateString'], r.loc[:,col].values, color='lightslategrey',edgecolor='black')
                    axs[plt_c].set_ylabel("# of Records ("+col+")")
                    axs[plt_c].set_xticklabels(r['DateString'], rotation='vertical')
                    plt.tight_layout
                else:              #If more than one row of plots (axs object has 2 indexes when >1 row)
                    axs[plt_r,plt_c].bar(r['DateString'], r.loc[:,col].values, color='lightslategrey',edgecolor='black')  
                    axs[plt_r,plt_c].set_ylabel("# of Observations ("+col+")")
                    axs[plt_r,plt_c].set_xticklabels(r['DateString'], rotation='vertical')


            else: 
                print('ERROR: '+col+' is not string, float, date or int data type. Cannot plot.')
        else:
            if rows == 1: #If one row of plots (axs object has only 1 index when 1 row)
                axs[plt_c].annotate('ERROR: '+col+' does not contain values.',xy=(.2,.5),xycoords='axes fraction')
            else:
                axs[plt_r,plt_c].annotate('ERROR: '+col+' does not contain values.',xy=(.2,.5),xycoords='axes fraction')
            
    plt.subplots_adjust(hspace = 0.3)


def compare_years_continuous(data, col, year1, year2, binwidth):
    """Compare two years of continuous variable data graphically. Works on only 1 column and 2 years of data.
    INPUTS:
    col = string, variable to examine
    data = pandas dataframe, entire dataframe not subsetted one
    year1 = string, ex. '2017'
    year2 = string ex. '2018'
    binwidth = int, width of each bin

    RETURNS: plots"""
    
    #Subset data
    subA = year_subset(data, year1, mute=True)
    subB = year_subset(data, year2, mute=True)
    
    #extract by column
    thA = subA.loc[:,col].values
    thB = subB.loc[:,col].values

    #Pad shorter dataset so they're the same length
    padlength = abs(len(thA)-len(thB))

    if len(thA)>len(thB):
        thB = np.pad(thB,(0,padlength),'constant')

    else:
        thA = np.pad(thA,(0,padlength),'constant')

    combined = np.stack((thA,thB),axis=1) #combine horizontally

    # Plot histogram
    fig, axs = plt.subplots(1, 1, figsize=(7, 7)) #Create figure
    max_bin = (math.ceil(np.nanmax(combined)/binwidth)*binwidth) #Find right most edge of bins
    cBins = np.arange(0, max_bin+binwidth, binwidth) #list of bins
    axs.hist(combined,bins=cBins,edgecolor='black', histtype='bar', color=['skyblue','lightslategrey'])
    axs.set_xticks(cBins) #Set ticks/labels as each bin
    
    #Add mean vertical lines
    axs.axvline(np.nanmean(thA),color='b')
    axs.axvline(np.nanmean(thB),color='black')
    
    #Add legend and labels
    axs.legend([year1+' Mean = '+str(round(np.nanmean(thA)))+'  STD = '+str(round(np.nanstd(thA))),year2+' Mean = '+str(round(np.nanmean(thB)))+'  STD = '+str(round(np.nanstd(thB))),year1,year2])
    axs.set_xlabel(col)
    axs.set_ylabel('Frequency')

def compare_years_catagorical(data, cols_to_plot, year1, year2):
    """Compare two years of catagorical variable data graphically. Works on list of columns and 2 years of data.
    INPUTS:
    cols_to_plot = list of strings of column names to analyze ex. ['Var_1','Var_2']
    data = pandas dataframe, entire dataframe not subsetted one
    year1 = string, ex. '2017'
    year2 = string, ex. '2018'

    RETURNS: plots"""
    
    #Create figure based on number of columns
    rows = math.ceil(len(cols_to_plot)/2)
    fig, axs = plt.subplots(rows, 2, figsize=(14, 7*rows))
    

    #Loop through each column and plot data
    for i in range(0,len(cols_to_plot)):
        col = cols_to_plot[i]

        #Subset data
        subA = year_subset(data, year1, mute=True)
        subB = year_subset(data, year2, mute=True)

        
        if (subA.loc[:,col].isna().all() == False)&(subB.loc[:,col].isna().all() == False): #check both subsets have data
            
            #Get index of subplot
            if i%2 == 0: #If even
                plt_c = 0      #Plot Column
                plt_r = int(i/2)  #Plot row
            if i%2 == 1: #If odd
                plt_c = 1       #Plot column
                plt_r = int(i/2-.5)   #Plot row
                
            #extract data by column and combine
            cA = pd.DataFrame(subA.loc[:,col].value_counts())
            cB = pd.DataFrame(subB.loc[:,col].value_counts())
            combined = cA.merge(cB, left_on=cA.index, right_on=cB.index,how='outer')

            #Add bars
            width = 0.35  # the width of the bars
            x = np.arange(len(combined.index)) #bar positions
            
            if rows == 1:      #If one row of plots (axs object has only 1 index when 1 row)
                axs[plt_c].bar(x - width/2, combined.iloc[:,1], width,edgecolor='black',color='skyblue')
                axs[plt_c].bar(x + width/2, combined.iloc[:,2], width,edgecolor='black',color='lightslategrey')

                #Add labels and legend
                axs[plt_c].set_xticks(x)
                axs[plt_c].set_xticklabels(combined.key_0.values.tolist(), rotation='vertical')
                axs[plt_c].set_ylabel('Count')
                axs[plt_c].legend([year1, year2])
                axs[plt_c].set_title(col)
            
            else:        #If more than 1 row of plots          
                axs[plt_r,plt_c].bar(x - width/2, combined.iloc[:,1], width,edgecolor='black',color='skyblue')
                axs[plt_r,plt_c].bar(x + width/2, combined.iloc[:,2], width,edgecolor='black',color='lightslategrey')

                #Add labels and legend
                axs[plt_r,plt_c].set_xticks(x)
                axs[plt_r,plt_c].set_xticklabels(combined.key_0.values.tolist(), rotation='vertical')
                axs[plt_r,plt_c].set_ylabel('Count')
                axs[plt_r,plt_c].legend([year1, year2])
                axs[plt_r,plt_c].set_title(col)
        
        else:
            print('ERROR: '+col+' does not contain values.')
    
    plt.subplots_adjust(hspace = 1) #add white space between subplots


def compare_surveyors(data, col, obs_var, drop):
    """Compares differences between observers for one variable
    INPUT: data = Pandas dataframe, subset or entire data
        col = string, variables of interest. Catagorical data only.
        obs_var = string, observer column name
        drop = integer 0 to 100, % threshold to drop observer from analysis. 
            If surveyor did not observe above this % of total observations they are dropped.
            
    RETURNS: two plots"""
    
    dataType = type(data.loc[data[col].notnull(),col].values[0])

    #Find # minimum observation needed to observe to qualify for analysis
    count_of_obs = data[col].count()
    drop_thresh = count_of_obs * (drop/100)
    print('Dropping surveyors with less than '+str(drop_thresh)+' observations\nThey did not make over '+str(drop)+'% of total observation')

    #Remove surveyors who made less that 20 observations
    logical = data.loc[data[col].notnull(),obs_var].value_counts() < drop_thresh
    to_drop = logical[logical == True].index.values
    print('Dropping Surveyors:', to_drop)
    subset = data[~data[obs_var].isin(to_drop)]

    #Tally and normalize observations by surveyor
    obs_tally = subset.groupby([obs_var])[col].value_counts()
    obs_total = subset.groupby([obs_var])[col].count()
    normalized = obs_tally / obs_total

    if (dataType==str): 
        #Plot Obs by Col
        catCount = data[col].nunique()
        colormap=cm.Paired(np.linspace(0,1,catCount))
        fig1, ax1 = plt.subplots(facecolor='white')
        ax1 = normalized.unstack().plot(kind='bar', figsize=(20,10), color=colormap, ax=ax1)
        

        ax1.set_ylabel("% of Surveyor's Observations")
        ax1.set_title('Check if general distribution of each surveyor is the same:')
        
        #Plot Col by Obs
        normalized = normalized.swaplevel(0,1).sort_index()
        colormap=cm.Set1(np.linspace(0,1,len(obs_total)))
        
        fig, ax = plt.subplots(facecolor='white')
        ax = normalized.unstack().plot(kind='bar', figsize=(20,10), color=colormap, ax=ax)
        leg_cols = obs_total.reset_index()
        legend_items = leg_cols.iloc[:,0].map(str) + ' N=' + leg_cols.iloc[:,1].map(str)
        ax.legend(legend_items)
        
        plt.rcParams['axes.facecolor']='white'
        ax.set_ylabel("% of Surveyor's Observations")
        ax.set_title('Check bars of each catagory are the same height for each surveyor:')
    
        
    elif (dataType == np.float64) | (dataType == np.int64):

        normalized = (pd.DataFrame(normalized))
        normalized.columns = ['Pct_Obs']
        normalized = normalized.reset_index()
        
        #display(normalized)
        groups = normalized.groupby(obs_var)
        
        fig, ax = plt.subplots(figsize=(15,10), facecolor='white')
        color=iter(cm.rainbow(np.linspace(0,1,8)))
            
        for name, group in groups:

            c=next(color)
            ax.plot(group[col], group.Pct_Obs, marker='o', linestyle='', label=name, c=c)
            
            z = np.polyfit(group[col], group.Pct_Obs, 4)
            
            xnew = np.linspace(group[col].min(),group[col].max(),300) #300 represents number of points to make between T.min and T.max
            ynew = np.polyval(z,xnew)
            
            ax.plot(xnew, ynew,c=c)

        ax.set_xlabel(col)
        ax.set_ylabel("% of Surveor's Observations")
        ax.legend()
        plt.show()




def QAQC_gps(data, rms_threshold):
    """Displays records where GPS spatial error is above threshold
    INPUTS:
    data = pandas dataframe
    rms_threshold = float, spatial error threshold, meters
    
    RETURNS: displays records that are above threshold"""
    
    logical = data.loc[:,'ESRIGNSS_AVG_H_RMS'] > rms_threshold
    
    if logical.any():
        print('GPS spatial error is above threshold in records below:')
        display(data.loc[logical,:])
    else:
        print('No entries above spatial error threshold')


def text_cleaner(text):
    """Removes spaces and converts line breaking space to normal space.
    INPUT: text = pandas series of strings
    OUTPUT: pandas series, cleaned"""
    a = text.str.replace('\xa0',' ')
    b = a.str.replace('  ',' ') #Double space replacement
    c = b.str.replace('   ',' ') #Triple space relacement
    final = c.str.strip()
    
    return final


def QAQC_whitespace(data, exclude_cols=None, objectID=None):
    """Looks for triple/double spaces, white space at end of string, and non breaking space '\xa0' 
    Does not work on columns with numbers. Do not need to check notes/comments/any field we type entries into. 
    
    INPUT: data = Pandas series
           exclude_cols = optional, list of strings, columns you do not want to check such as comments.
           objectID = optional, string in list, name of object ID column, if not supplied function will try to find it.
    OUTPUT: displays which entries has issues"""
    #Check inputs
    if check_inputs(data=[data]):

        #Find Object ID column name if not supplied
        if not objectID:
            objectID = [col for col in data.columns.values if 'object' in col.lower()]

            #Object ID error handling
            if len(objectID) == 1:
                print('Name of Object ID Column: ',objectID)
            elif len(objectID) < 1:
                print("ERROR: Could not find Object ID Column, does it contain 'object' in the name?")
                return 
            else:
                print('ERROR: Found multiple Object IDs')
                return 
            
       

        #Get list of columns that are data type == object (strings)
        tCols = (data.columns.values[data.dtypes=='object'])

        #Remove excluded columns
        if exclude_cols:
            tCols = list(set(tCols) - set(exclude_cols))

        #Loop through list of test columns
        for col in tCols:
            print('\n\nAnalysis for:', col)
            text = data[col]
            nonBreakL = text.str.contains('\xa0', regex=False, na=False)
            twoSpaceL = text.str.contains('  ', regex=False, na=False)
            threeSpaceL = text.str.contains('   ', regex=False, na=False)
            trailSpace = text.str.endswith(' ', na=False)
            
            #Add column name to object ID list
            displaycols = objectID + [col]
            
            if nonBreakL.any():
                print(r'Found \xa0 characters')
                display(data.loc[nonBreakL, displaycols])
            else:
                print(r'No \xa0 characters found')

            if twoSpaceL.any():
                print('Found double space')
                display(data.loc[twoSpaceL, displaycols])
            else:
                print('No double spaces found')

            if threeSpaceL.any():
                print('Found triple space')
                display(data.loc[threeSpaceL, displaycols])
            else:
                print('No triple spaces found')

            if trailSpace.any():
                print('Trailing white space found')
                display(data.loc[trailSpace, displaycols])
            else:
                print('No trailing white space found')
                
                


def validate_domains(feature_service):
    """Compares data against domains and displays data entries that are not within the domain. It checks all tables and layers
    INPUT: feature service
    OUTPUT: Displays mismatches"""
    pd.set_option('display.max_colwidth', 0)
    
    def compare(layersTables):
    
        for lt in layersTables: #Loop through each table or layer
            print('\n\nTable/Layer: ',lt.properties.name )

            found = False #Flag to track if a mismatch is found
            
            results = pd.DataFrame(columns=['Field','Data'])
            if lt.query(return_count_only=True) > 0: #If there are any records
                
                data = lt.query().sdf #Convert data to pandas dataframe
                for field in lt.properties.fields:    #Loop through fields

                    if field.domain:         #If domain exists
                        domain_codes = set()   #Make set of coded domain values
                        for entry in field.domain.codedValues:
                            domain_codes.add(entry.code)
                        domain_codes.add(None) #Adds None as an option so that blank entries are ignored

                        records = set(data[field.name]) #Make set of data
                        result = records - domain_codes #subtract sets to find the records that are not in domain.
                        
                        if result:#If mismatch, add to results dataframe
                            result_df = pd.DataFrame({'Field': [field.name], 'Data': [result]})
                            results = results.append(result_df)
                            found = True
                
                if found: #If there were mismatches, display results dataframe
                    print('The following entries are not within the domain:')
                    display(results)
                else:
                    print('No Mismatches Found')
            else:
                print('No records/features\n')
                            
    compare(feature_service.tables)#do comparison for tables
    compare(feature_service.layers)#do comparison for layers
    
def QAQC_attachment_filenames(feature_service):
    """Examines filenames of photo attachments. It displays any entries where 'photo' is contained in the filename, this is likely a mislabeled photo
    INPUT: feature service
    RETURNS: displays messages"""

    lyrs_count = len(feature_service.layers)
    tbls_count = len(feature_service.tables)
    
    #Loop through layers
    for l in range(0,lyrs_count):
        lyr = feature_service.layers[l]
        examine_lyrtbl_att(lyr)

    #loop through tables
    for t in range(0,tbls_count):
        tbl = feature_service.tables[t]
        examine_lyrtbl_att(tbl)
        

def examine_lyrtbl_att(lyrtbl):
    #Called by QAQC_attachment_filenames. Takes in a layer or table and looks at attachment file names and displays ones that contain "photo" or "attachment" in file name.
    i = 0
    foundflag = False
    record_count = lyrtbl.query(return_count_only=True)
    
    if lyrtbl.properties['hasAttachments']:
        
        #Get object ID field name
        objIDfield = lyrtbl.properties['objectIdField']
        fts = lyrtbl.query(out_fields=objIDfield)
        
        
        #Loop through each feature
        for feature in fts.features:
            objectID = feature.attributes[objIDfield]
            att_list = lyrtbl.attachments.get_list(oid=objectID)
            
            #Loop through each attachment on feature
            for att in att_list:
                att_name = att['name']
                if ('photo' in att_name.lower()) or ('attachment' in att_name.lower()):
                    #If first mismatch find...
                    if foundflag==False:
                        print('Found mislabeled photo attachement(s):')
                        print(lyrtbl.properties['name'])
                        foundflag=True
                    print('Object ID:',objectID,' Current Name: ', att_name)
            i+=1
        if foundflag==False:
            print('\nNo mislabeled photos found for layer/table',lyrtbl.properties['name'])
    else:
        print('\nNo attachments for layer/table', lyrtbl.properties['name'])
        
        
def find_entry(data, check_cols, search, objectID = 'OBJECTID'):
    """Checks target columns in data for search terms
    INPUTS: data = pandas dataframe
            check_cols = list of strings, columns to search in, ex. ['Var_1','Var_2']
            search = list of string, search words, ex. ['Yes','No']
            objectID = default: 'OBJECTID', string, name of identifier column
            
    RETURNS: pandas dataframe of entries where words are found
        also displays found records"""

    for col in check_cols:
        logical = data[col].isin(search)
        
        if logical.any():
            print("Found "+str(search)[1:-1]+" in "+col+':')
            found = data.loc[logical,:]
            display(found[[objectID,col]])
        else:
            print("Did not find "+str(search)[1:-1]+" in",col)
            found = []
            
    return found

    