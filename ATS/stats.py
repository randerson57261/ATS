import statsmodels.api as sm
from patsy import dmatrices
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def confidence_interval(alive, dead):

    if not np.isnan(alive):
        a = np.repeat(1, alive)
        d = np.repeat(0, dead)
        ad = (np.concatenate((a,d)))
    
        means = []
        i = 0
        while i < 1000:
            ad_mean = np.random.choice(ad, len(ad)).mean()
            means.append(ad_mean)
            i += 1

        means_np = np.asarray(means)

        lower = np.percentile(means_np, 2.5)
        upper = np.percentile(means_np, 97.5)

        return [lower,upper]
    else:
        return [None, None]
    
    
def glm_binomial(model_data, model_formula):
    """Fits a generalized linear model with binomial errors to the data
    INPUTS:
    model_data = dataframe, contains 3 columns (Year/Category, Alive/Present Count, Dead/Absent Count)
    model_formula = string, R format of model corresponding to column names. Ex: 'Alive + Dead ~ Year'
    
    Returns a model object to be used in plotting function"""
    
    #Convert data format
    y, x = dmatrices(model_formula, model_data, return_type='dataframe')

    #Initiate model object
    binom_model = sm.GLM(y, x, family=sm.families.Binomial())

    binom_model_results = binom_model.fit()

    return binom_model_results

def glm_binomial_plot(binom_model_results, model_data, end_year, y_label='%', y_max=100, fig_size = (6,6)):
    """Creates a plot of the model result from the function 'glm_binomial' with real world values, predicted values, and 95% CI interval cone
    INPUTS:
    binomial_model = model object, output of glm_binomial function
    model_data = dataframe, contains same data use to create model
    end_year = int, end date of plot
    y_label = string, y label of plot
    y_max = int, y max of plot"""
    
    #Plot model
    # Generate new x values to feed into model, and new intercepts (always=1)
    new_x = np.arange(start= 2018, stop= end_year+1, step = 1)
    new_intercept = np.repeat(1,len(new_x))

    #Combine new_x and new_intercept into dataframe
    new_predictors = pd.DataFrame(zip(new_intercept, new_x))

    #Predict
    new_y = binom_model_results.predict(new_predictors)
    predict_results = binom_model_results.get_prediction(new_predictors)

    #Add pct column
    model_data['Pct'] = model_data.Success / (model_data.Success + model_data.Fail)

    #Add new x column to dataframe
    model_data = pd.concat([model_data,pd.DataFrame(new_x)], ignore_index=True, axis=1)
    model_data.columns = ['Year','Success','Fail','Pct','Plot_Year_Predict']


    #plot, adjust graphics
    fig, ax = plt.subplots(figsize=fig_size)
    x_pos = np.arange(len(model_data))
    ax.bar(x_pos, model_data.Pct*100, align='center', color=(0,130/255,196/255,1))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(np.arange(2018,end_year+1,1))
    ax.set_ylabel(y_label)
    ax.set_ylim(0,y_max)

    #Get Confidence interval and create CI cone verticy array
    CI = predict_results.conf_int()
    CI_lower = CI[:,0]*100
    CI_upper = CI[:,1]*100
    CI_upper_flip = np.flip(CI_upper)
    cone_y = np.concatenate((CI_lower, CI_upper_flip))
    cone_x = np.concatenate((x_pos, np.flip(x_pos)))
    cone_verts = np.stack((cone_x, cone_y),axis=1)

    #Add prediction of model line
    ax.plot(x_pos, new_y*100, color='black', linestyle='--')

    #Add CI cone
    CI_cone = Polygon(cone_verts, facecolor='0', alpha=.3)
    ax.add_patch(CI_cone)