#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 16:11:39 2022

@author: madeleip
"""


from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt

from sklearn.inspection import permutation_importance
import pandas as pd 
from sklearn.metrics import mean_squared_error
from scipy import stats
from scipy.stats import gaussian_kde
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# import mpl_scatter_density # adds projection='scatter_density'
from sklearn.model_selection import RandomizedSearchCV

def RandomForestDataSplit(features):

    #save feature names 
    feature_list = list(features.columns)
    dependent_name = feature_list[0]
    feature_list=feature_list[1:len(feature_list)]
    
    #convert to array 
    features=np.array(features)
    print(len(features))
    print(len(features)*0.5)

    #Remove missing from each row 
    features=features[~np.isnan(features).any(axis=1)]
    
    #Get predictors and dependent var
    y=features[:,0]
    X=features[:,1:len(features)]

    # Split the data into training and test sets (80% training, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Further split the training set into training and validation sets (80% training, 20% validation)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


    #Coordinates of Prediction  
    idlon=feature_list.index('X')
    idlat=feature_list.index('Y')
   
    ypredlon=X_test[:,idlon]
    ypredlat=X_test[:,idlat]


  

    return X_train, X_test, X_val, y_train, y_test, y_val


def RandomForestTrain(features, X_train, X_test, X_val, y_train, y_test, y_val):

    # Train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model on the validation set
    y_val_pred = model.predict(X_val)

    
    #MSE
    mse=mean_squared_error(y_val_pred, y_val)
    
    #Get Random Forest R2 for test set
    test_score = r2_score(y_val_pred, y_val)
    
    print(f'Val Set R-2 score: {test_score:>5.3}')
    print(f'Val Set MSE: {mse:>5.3}')

    return model


def RandomForestValTune(X_val, y_val, model):


    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False]
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator = model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

    # Fit the model
    grid_search.fit(X_val, y_val)

    # Get the best parameters
    best_params = grid_search.best_params_

    print(f'Best parameters: {best_params}')
    
    return best_params



def RandomForestBestModel(X_train, X_test, X_val, y_train, y_test, y_val, best_params):
    """ Combine train and validation data with 
    best params, and then test on test set"""
    
    print('combine datas val and train')
    X_train_combined = np.concatenate((X_train, X_val), axis=0) 
    y_train_combined = np.concatenate((y_train, y_val), axis=0)

    #Re - Train the Random Forest model
    print('running model with best params...')
    best_model = RandomForestRegressor( n_estimators=best_params['n_estimators'], 
                               max_depth=best_params['max_depth'], 
                               min_samples_split=best_params['min_samples_split'], 
                               min_samples_leaf=best_params['min_samples_leaf'], 
                               max_features=best_params['max_features'], 
                               bootstrap=best_params['bootstrap'], 
                               random_state=42, 
                               oob_score=True )
    
    best_model.fit(X_train_combined, y_train_combined)

    #Test on Test Data

    y_test_pred = best_model.predict(X_test) 
    
    #MSE
    mse=mean_squared_error(y_test_pred, y_test)
    
    #Get Random Forest R2 for test set
    test_score = r2_score(y_test_pred, y_test)
    
    print(f'Val Set R-2 score: {test_score:>5.3}')
    print(f'Val Set MSE: {mse:>5.3}')
    

    return y_test_pred, X_test




###### - OLDEr

def RandomForestTune(features, N_train, N_validation, N_test, name):
    
    #save feature names 
    feature_list = list(features.columns)
    dependent_name = feature_list[0]
    feature_list=feature_list[1:len(feature_list)]
    
    #convert to array 
    features=np.array(features)
    print(len(features))
    print(len(features)*0.5)

    #Remove missing from each row 
    features=features[~np.isnan(features).any(axis=1)]
    
    #Get predictors and dependent var
    y=features[:,0]
    X=features[:,1:len(features)]
    
    #Split into test and training set - random selection #Test size is proportion saved for test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = N_test, train_size = N_train, random_state = 42)
    

    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False]
    }

    # Initialize the RandomForestRegressor
    rf = RandomForestRegressor(random_state=42, oob_score=True)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

    # Fit the model
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_

    print(f'Best parameters: {best_params}')
    
    return best_params





def RandomForestRegression(features, N_train, N_test, name, best_params):
    """ Create random forest regression model
        
        Input: 
        Features [df]
        N number of points in training model (as a proportion)
        name: 
        Tune = 0 (no) or 1 (yes)
            
        output: R2, Var Weights of Importance 
        """

    #save feature names 
    feature_list = list(features.columns)
    dependent_name = feature_list[0]
    feature_list=feature_list[1:len(feature_list)]
    
    #convert to array 
    features=np.array(features)
    print(len(features))
    print(len(features)*0.5)
    #Remove missing from each row 
    features=features[~np.isnan(features).any(axis=1)]
     
    #Get predictors and dependent var
    y=features[:,0]
    X=features[:,1:len(features)]
    
    #Split into test and training set - random selection 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = N_test, train_size = N_train)
   
    #Coordinates of Prediction  
    idlon=feature_list.index('X')
    idlat=feature_list.index('Y')
   
    ypredlon=X_test[:,idlon]
    ypredlat=X_test[:,idlat]
    
    X_test=X_test[:,0:idlon]
    X_train=X_train[:,0:idlon]
    
    # Fitting Random Forest Regression to the dataset
    rng = np.random.RandomState(0)
    
    #Run using best-params
    rf = RandomForestRegressor( n_estimators=best_params['n_estimators'], 
                               max_depth=best_params['max_depth'], 
                               min_samples_split=best_params['min_samples_split'], 
                               min_samples_leaf=best_params['min_samples_leaf'], 
                               max_features=best_params['max_features'], 
                               bootstrap=best_params['bootstrap'], 
                               random_state=42, 
                               oob_score=True )
    
    #Run RF model using the training data
    rf.fit(X_train, y_train)               

    #Generate prediction with Test Data   
    y_pred = rf.predict(X_test)            
     
    test_accuracy=rf.score(X_train, y_train)
    train_accuracy =rf.score(X_test, y_test)
    print (f'Accuracy test data: {test_accuracy:>5.3}')#Test accuracy of training data
    print (f'Accuracy train data: {train_accuracy:>5.3}')#test accuracy of test data 
    #If the training accuracy is so high and the testing is not as close, it’s safe to say model is overfit on training...
    
    #MSE 
    mse=mean_squared_error(y_test, y_pred)
    
    #Get Random Forest R2 for test set
    test_score = r2_score(y_test, y_pred)
    print(f'Out-of-bag R-2 score estimate: {rf.oob_score_:>5.3}')
    print(f'Test data R-2 score: {test_score:>5.3}')
    print(f'MSE: {mse:>5.3}')
    

   
    #Feature Importance from Permutation
    print('Generating feature importnace from permutation...')
    result=permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=1)
    sorted_idx = result.importances_mean.argsort()
    Xlabel = pd.DataFrame(columns=feature_list)
 
        
    fig, ax = plt.subplots()
    ax.boxplot(
    result.importances[sorted_idx].T, vert=False, labels=Xlabel.columns[sorted_idx])
    ax.set_xlabel("Permutation Importances")
    ax.set_title(name)
    ax.set_xlim(0,1)
    font = {'family' : 'sans',
        'weight' : 'bold',
        'size'   : 12}

    plt.rc('font', **font)
    fig.tight_layout()
    plt.show()
    
    
    
    #get list of feature importances 
    
    
    importances = list(rf.feature_importances_)
    feature_importances=[(feature, round(importance,2)) for feature,importance in zip(feature_list,importances)]
     #sort the feature importances 
    feature_importances=sorted(feature_importances,key=lambda x: x[1], reverse =True)
     #print most important features from most to least ... 
    print('Generating feature importnace from impurity')
    [print('Variable: {:20} IMportance: {}'.format(*pair)) for pair in feature_importances];
    

    
    
    return test_score, mse,feature_importances, feature_list,X_train, X_test, y_train, y_test,y_pred,ypredlon,ypredlat,rf
   





def scatter_density_plot(y_test, y_pred):
    """Generate scatter density"""
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_test,y_pred)
    line = slope*y_test+intercept
  
    #Figure
    
    
    
    from matplotlib.colors import LinearSegmentedColormap

    # "Viridis-like" colormap with white background
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'),
        (1e-20, '#440053'),
        (0.2, '#404388'),
        (0.4, '#2a788e'),
        (0.6, '#21a784'),
        (0.8, '#78d151'),
        (1, '#fde624'),
        ], N=256)
    
    fig = plt.figure()
   
    ax = fig.add_subplot(1, 1, 1, projection ='scatter_density')    
    ax.grid(False)
    ax.patch.set_edgecolor('black')  
    ax.patch.set_linewidth(4) 
    density = ax.scatter_density(y_test/1000, y_pred/1000, vmax = 200, cmap = white_viridis)
    fig.colorbar(density, label='Number of points per pixel')
    

    
     # # #Visualizations 
    plt.plot(y_test/1000, line/1000, 'r', label='y={:.2f}x+{:.2f}'.format(slope,intercept))
    
    plt.plot(y_test/1000, y_test/1000, 'k:',label='1:1')
  
    #Scatter density
    # Calculate the point density
    #xy = np.vstack([y_test,y_pred]) 
    #z = gaussian_kde(xy)(xy)
  
    #plt.scatter(y_test,y_pred)#Scatter Density 
    
       
    
    plt.xlabel('observed')
    plt.ylabel('predicted')
   # plt.title(dependent_name)
    
    plt.xlim(-0.700,1.100)
    plt.ylim(-0.700,1.100)
    plt.annotate("r-squared = {:.2f}".format(r2_score(y_test,y_pred)), (np.max(line)*0.25, np.min(line)))
   
    plt.legend(fontsize=14)
    plt.show()
    #End 

    

def ApplyRandomForestRegression(features,N_train, N_test,name,rf):
    #Use pre-existing RF model to run on a different region
    #input: Features [includes dependent and predictors vars where first column 
    #is dependent ] format:dataframe
    #N - number of points in training model (as a proportion)
    #name: fire name
    #rf : the random forest model already created 
        
    #output: R2, Var Weights of Importance 

    #save feature names 
    feature_list = list(features.columns)
    dependent_name = feature_list[0]
    feature_list=feature_list[1:len(feature_list)]
    
    #convert to array 
    features=np.array(features)
    print(len(features))
    print(len(features)*N_train)
    #Remove missing from each row 
    features=features[~np.isnan(features).any(axis=1)]
    

    #Get predictors and dependent var
    y=features[:,0]
    X=features[:,1:len(features)]
    
    #Split into test and training set - random selection #Test size is proportion saved for test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = N_test, train_size = N_train)
    
    #Coordinates of Prediction 
    
    idlon=feature_list.index('X')
    idlat=feature_list.index('Y')

   
    ypredlon=X_test[:,idlon]
    ypredlat=X_test[:,idlat]
    
    X_test=X_test[:,0:idlon]
    X_train=X_train[:,0:idlon]
    
    
    y_pred = rf.predict(X_test)             #Generate prediction with Test Data and loaded model
     
    #MSE 
    mse = mean_squared_error(y_test, y_pred)
    test_score = r2_score(y_test, y_pred)
    print(f'Out-of-bag R-2 score estimate: {rf.oob_score_:>5.3}')
    print(f'Test data R-2 score: {test_score:>5.3}')
    print(f'MSE: {mse:>5.3}')
    

    #==========================================================================
    #Feature Importance from Permutation
    #==========================================================================
    print('Generating feature importnace from permutation...')
    # #Get list of PERMUTATION feature importances 
    result=permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=1)
    sorted_idx = result.importances_mean.argsort()
    Xlabel = pd.DataFrame(columns=feature_list)
 
        
    fig, ax = plt.subplots()
    ax.boxplot(
    result.importances[sorted_idx].T, vert=False, labels=Xlabel.columns[sorted_idx])
    ax.set_xlabel("Permutation Importances")
    ax.set_title("Predicted Fire Importance")
    ax.set_xlim(0,1)
    font = {'family' : 'sans',
        'weight' : 'bold',
        'size'   : 12}

    plt.rc('font', **font)
    fig.tight_layout()
    plt.show()
    
    
    feature_importances = result.importances
   
    
       
    return test_score, mse,feature_importances, feature_list,X_train, X_test, y_train, y_test,y_pred,ypredlon,ypredlat,rf

def TuneRF(X_train,y_train,X_test,y_test):
    """
        @author: madeleip
    Tunes random forest model 'rf'
    Input: training and test dat
    Output: rf_tune
    """

    
    #==========================================================================
    #Tuning MOdel 
    #==========================================================================
   

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
    n_estimators = [int(x) for x in np.linspace(start = 10, stop = 500, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    #rndom state 
    rng = np.random.RandomState(0)
  
    rf_tune = RandomForestRegressor(n_estimators = 50, random_state = rng,oob_score=True)
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_tune = RandomizedSearchCV(estimator = rf_tune, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    # Fit the random search model
    rf_tune.fit(X_train, y_train)                #Run RF model using the training data 
    y_pred = rf_tune.predict(X_test)             #Generate prediction with Test Data 
    
    
    #Test the tuned model 
    #MSE 
    mse=mean_squared_error(y_test, y_pred)
    
    
    #Get Random Forest R2 for test set
    #Get R2 using the Test Data and Prediction 
    #Coefficient of determination 
    test_score = r2_score(y_test, y_pred)

       

    
    return test_score, mse,y_pred,rf_tune


def RandomForestClassification(features,N_train, N_test,name, rf_type):
    #Create random forest classificiation 
    #input: Features [includes dependent and predictors vars where first column 
    #is dependent ] format:dataframe
    #N - number of points in training model (as a proportion)
    #name: 
    #Tune = 0 (no) or 1 (yes)
        
    #output: R2, Var Weights of Importance 

    #save feature names 
    feature_list = list(features.columns)
    dependent_name = feature_list[0]
    feature_list=feature_list[1:len(feature_list)]
    
    #convert to array 
    features=np.array(features)
    print(len(features))
    print(len(features)*0.5)
    #Remove missing from each row 
    features=features[~np.isnan(features).any(axis=1)]
    
   
    #Get predictors and dependent var
    y=features[:,0]
    X=features[:,1:len(features)]
    
    #Split into test and training set - random selection 
    #Test size is proportion saved for test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = N_test, train_size = N_train)
    #print(len(X_train))
    
    
    
    #>>>>>>>>>> Comment here to include lat lon
    
    #Coordinates of Prediction 
    
    idlon=feature_list.index('X')
    idlat=feature_list.index('Y')

   
    ypredlon=X_test[:,idlon]
    ypredlat=X_test[:,idlat]
    
    X_test=X_test[:,0:idlon]
    X_train=X_train[:,0:idlon]
    

    #>>>>>>>>>> 
    
    
    #>>>>>>>>>> 
    # Fitting Random Forest Regression to the dataset
    #n_estimator = decision trees
    #>>>>>>>>>> 
    
    #rndom state 
    clf=RandomForestClassifier()
    clf.fit(X_train, y_train)               #Run RF classification model using the training data 
    
    y_pred = clf.predict(X_test)             #Generate prediction with Test Data 
     
    test_accuracy=clf.score(X_train, y_train)
    train_accuracy =clf.score(X_test, y_test)
    print (f'Accuracy test data: {test_accuracy:>5.3}')#Test accuracy of training data
    print (f'Accuracy train data: {train_accuracy:>5.3}')#test accuracy of test data 
    #If the training accuracy is so high and the testing is not as close, it’s safe to say model is overfit on training...
    
    #MSE 
    mse=mean_squared_error(y_test, y_pred)
    #print(mse)
    #print(np.var(y_test))
    
    
    #Get Random Forest R2 for test set
    test_score = r2_score(y_test, y_pred)
    print(f'Test data R-2 score: {test_score:>5.3}')
    
       
    
     
    
    #==========================================================================
    #Feature Importance from Impurity Index
    #==========================================================================
   
    #get list of feature importances 
    feature_importances =[]
    
    
    #.
    #.
    #.
    #.
    # importances = list(clf.feature_importances_)
    # feature_importances=[(feature, round(importance,2)) for feature,importance in zip(feature_list,importances)]
    #  #sort the feature importances 
    # feature_importances=sorted(feature_importances,key=lambda x: x[1], reverse =False)
    #  #print most important features from most to least ... 
    # print('Generating feature importnace from impurity')
    # [print('Variable: {:20} IMportance: {}'.format(*pair)) for pair in feature_importances];
    
    # lst2 = [item[0] for item in feature_importances]
    # lst3 = np.array([item[1] for item in feature_importances])
   # .
   
   
   
     
    # fig, ax = plt.subplots()
    # ax.barh(range(0,lst3.size),
    # lst3)
    # plt.yticks(range(0,lst3.size), lst2, rotation=0)
    # ax.set_xlabel("Permutation Importances")
    # ax.set_title(name)
    # ax.set_xlim(0,1)
    # font = {'family' : 'sans',
    #     'weight' : 'bold',
    #     'size'   : 12}
  
    # plt.rc('font', **font)
    # fig.tight_layout()
    # plt.show()
  
   
   
   
   
    #.
    #.
    #.
    
    #==========================================================================
    # 
    #==========================================================================

    
    
    
    
  

    #remove outlier
    y_test=np.where(y_test<-np.std(y_test)*3,np.nan,y_test)
    y_pred=np.where(y_pred<-np.std(y_pred)*3,np.nan,y_pred)
    
    finiteYmask = np.isfinite(y_pred)
    
    Yclean = y_pred[finiteYmask]
    Xclean = y_test[finiteYmask]
    
    finiteXmask = np.isfinite(y_test)
    
    Yclean = y_pred[finiteXmask]
    Xclean = y_test[finiteXmask]
    
    y_test=Xclean
    y_pred=Yclean
    
  
    ypredlon=ypredlon[finiteYmask]
    ypredlon=ypredlon[finiteXmask]
    
    ypredlat=ypredlat[finiteYmask]
    ypredlat=ypredlat[finiteXmask]
    
    if rf_type == 'classification':
  
      id = np.where(y_test == 1) #Points in OBS classified as burn
      id_noburn = np.where(y_test == 0) #Points in OBS classified as no burn
     
      id_pred = np.where(y_pred == 1) #Points in prediction classified as burn 
      id_pred_noburn = np.where(y_pred == 0)#Points in prediction classified as no burn 
      
      
      
      # Percent of predicted burn points accurately classified
      correct = np.squeeze(np.array(np.where(y_test[id]==y_pred[id])))
      per_acc = (len(correct) / len(y_test[id])) * 100
      
      
      print(f'Percent Accurately Classified as Burn: {per_acc:>5.3}')
      
      # Percent of all points (burn/no burn) accurately classified
      
      correct = np.squeeze(np.array(np.where(y_test==y_pred)))
      per_acc = (len(correct) / len(y_test)) * 100
      
      
      print(f'Percent Accurately Classified: {per_acc:>5.3}')
      
      
      
      
      # Percent of burn omission error 
      incorrect = np.squeeze(np.array(np.where(y_test[id] != y_pred[id])))
      per_acc = (len(incorrect) / len(y_test[id])) * 100
      
      
      print(f'Percent Omission: {per_acc:>5.3}')
      
      # Percent of False Positive (comission) 
      incorrect = np.squeeze(np.array(np.where(y_test[id_pred] != y_pred[id_pred])))
      per_acc = (len(incorrect) / len(y_test[id_pred])) * 100
      
      
      print(f'False Positive / Error of Commision: {per_acc:>5.3}')
    
  
    
      

      
  
    
    
    return test_score, mse,feature_importances, feature_list,X_train, X_test, y_train, y_test,y_pred,ypredlon,ypredlat,clf


def ApplyRandomForestClassification(features,N_train, N_test,name,clf,rf_type):
    #Create random forest classificiation 
    #input: Features [includes dependent and predictors vars where first column 
    #is dependent ] format:dataframe
    #N - number of points in training model (as a proportion)
    #name: 
    #Tune = 0 (no) or 1 (yes)
        
    #output: R2, Var Weights of Importance 

    #save feature names 
    feature_list = list(features.columns)
    dependent_name = feature_list[0]
    feature_list=feature_list[1:len(feature_list)]
    
    #convert to array 
    features=np.array(features)
    print(len(features))
    print(len(features)*0.5)
    #Remove missing from each row 
    features=features[~np.isnan(features).any(axis=1)]
    
   
    #Get predictors and dependent var
    y=features[:,0]
    X=features[:,1:len(features)]
    
    #Split into test and training set - random selection 
    #Test size is proportion saved for test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = N_test, train_size = N_train)
    #print(len(X_train))
    
    
    
    #>>>>>>>>>> Comment here to include lat lon
    
    #Coordinates of Prediction 
    
    idlon=feature_list.index('X')
    idlat=feature_list.index('Y')

   
    ypredlon=X_test[:,idlon]
    ypredlat=X_test[:,idlat]
    
    X_test=X_test[:,0:idlon]
    X_train=X_train[:,0:idlon]
    

    #>>>>>>>>>> 
    
    
    #>>>>>>>>>> 
    # Fitting Random Forest Regression to the dataset
    #n_estimator = decision trees
    #>>>>>>>>>> 
    
   
    #predict with model     
    y_pred = clf.predict(X_test)             #Generate prediction with Test Data 
     
    test_accuracy=clf.score(X_train, y_train)
    train_accuracy =clf.score(X_test, y_test)
    print (f'Accuracy test data: {test_accuracy:>5.3}')#Test accuracy of training data
    print (f'Accuracy train data: {train_accuracy:>5.3}')#test accuracy of test data 
    #If the training accuracy is so high and the testing is not as close, it’s safe to say model is overfit on training...
    
    #MSE 
    mse=mean_squared_error(y_test, y_pred)
     
    
    #Get Random Forest R2 for test set
    test_score = r2_score(y_test, y_pred)
    #print(f'Test data R-2 score: {test_score:>5.3}')
    
       
    
     
    
    #==========================================================================

    #==========================================================================
    #Feature Importance from Impurity Index
    #==========================================================================
   
       
    result=permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=1)
    sorted_idx = result.importances_mean.argsort()
    Xlabel = pd.DataFrame(columns=feature_list)
 
   
    fig, ax = plt.subplots()
    ax.boxplot(
    result.importances[sorted_idx].T, vert=False, labels=Xlabel.columns[sorted_idx])
    ax.set_xlabel("Permutation Importances")
    ax.set_title("predicted")
   
    font = {'family' : 'sans',
        'weight' : 'bold',
        'size'   : 12}

    plt.rc('font', **font)
    fig.tight_layout()
    plt.show()
    
    
    feature_importances = result.importances
    #####
    

    #remove outlier
    # y_test=np.where(y_test<-np.std(y_test)*3,np.nan,y_test)
    # y_pred=np.where(y_pred<-np.std(y_pred)*3,np.nan,y_pred)
    
    # finiteYmask = np.isfinite(y_pred)
    
    # Yclean = y_pred[finiteYmask]
    # Xclean = y_test[finiteYmask]
    
    # finiteXmask = np.isfinite(y_test)
    
    # Yclean = y_pred[finiteXmask]
    # Xclean = y_test[finiteXmask]
    
    # y_test=Xclean
    # y_pred=Yclean
    
  
    # ypredlon=ypredlon[finiteYmask]
    # ypredlon=ypredlon[finiteXmask]
    
    # ypredlat=ypredlat[finiteYmask]
    # ypredlat=ypredlat[finiteXmask]
    
     
    # #regression part
    
    # slope, intercept, r_value, p_value, std_err = stats.linregress(y_test,y_pred)
   
    # print(f'Test data Lin Regress R2 score: {r_value**2:>5.3}')
   
  
    
 
    # Percent Accurately Classified
    # Percent BURN accurately cassified
    
    
    if rf_type == 'classification':
    
        id = np.where(y_test == 1) #Points in OBS classified as burn
        id_noburn = np.where(y_test == 0) #Points in OBS classified as no burn
       
        id_pred = np.where(y_pred == 1) #Points in prediction classified as burn 
        id_pred_noburn = np.where(y_pred == 0)#Points in prediction classified as no burn 
        
        
        
        # Percent of predicted burn points accurately classified
        correct = np.squeeze(np.array(np.where(y_test[id]==y_pred[id])))
        per_acc = (len(correct) / len(y_test[id])) * 100
        
        
        print(f'Percent Accurately Classified as Burn: {per_acc:>5.3}')
        
        # Percent of all points (burn/no burn) accurately classified
        
        correct = np.squeeze(np.array(np.where(y_test==y_pred)))
        per_acc = (len(correct) / len(y_test)) * 100
        
        
        print(f'Percent Accurately Classified: {per_acc:>5.3}')
        
        
        
        
        # Percent of burn omission error 
        incorrect = np.squeeze(np.array(np.where(y_test[id] != y_pred[id])))
        per_acc = (len(incorrect) / len(y_test[id])) * 100
        
        
        print(f'Percent Omission: {per_acc:>5.3}')
        
        # Percent of False Positive (comission) 
        incorrect = np.squeeze(np.array(np.where(y_test[id_pred] != y_pred[id_pred])))
        per_acc = (len(incorrect) / len(y_test[id_pred])) * 100
        
        
        print(f'False Positive / Error of Commision: {per_acc:>5.3}')
        
    
    return test_score, mse,feature_importances, feature_list,X_train, X_test, y_train, y_test,y_pred,ypredlon,ypredlat,clf


def RandomForestRegression_R2(features,N_train, N_test,name):
    #Create random forest regression model
    #Only returns R2
    
    #save feature names 
    feature_list = list(features.columns)
    dependent_name = feature_list[0]
    feature_list=feature_list[1:len(feature_list)]
    
    #convert to array 
    features=np.array(features)
    print(len(features))
    print(len(features)*0.5)
    #Remove missing from each row 
    features=features[~np.isnan(features).any(axis=1)]
    
    
    
    
    
    
    #>>>>>>>>>>
    
  
    
     
    #Get predictors and dependent var
    y=features[:,0]
    X=features[:,1:len(features)]
    
    
    #Split into test and training set - random selection 
    #Test size is proportion saved for test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = N_test, train_size = N_train)
    #print(len(X_train))
    
    
    
    #>>>>>>>>>> Comment here to include lat lon
    
    #Coordinates of Prediction 
    
    idlon=feature_list.index('X')
    idlat=feature_list.index('Y')

   
    ypredlon=X_test[:,idlon]
    ypredlat=X_test[:,idlat]
    
    X_test=X_test[:,0:idlon]
    X_train=X_train[:,0:idlon]
    
    #>>>>>>>>>> 
    
    
    
    # Fitting Random Forest Regression to the dataset
    #n_estimator = # of decision trees
    #rndom state 
    rng = np.random.RandomState(0)
    
    rf = RandomForestRegressor(n_estimators = 50, random_state = rng,oob_score=True)
    rf.fit(X_train, y_train)                #Run RF model using the training data 
    y_pred = rf.predict(X_test)             #Generate prediction with Test Data 
     
    test_accuracy=rf.score(X_train, y_train)
    train_accuracy =rf.score(X_test, y_test)
   # print (f'Accuracy test data: {test_accuracy:>5.3}')#Test accuracy of training data
    #print (f'Accuracy train data: {train_accuracy:>5.3}')#test accuracy of test data 
    #If the training accuracy is so high and the testing is not as close, it’s safe to say model is overfit on training...
    
    #MSE 
    mse=mean_squared_error(y_test, y_pred)
    #print(mse)
    #print(np.var(y_test))
    
    
    #Get Random Forest R2 for test set
    test_score = r2_score(y_test, y_pred)
   # print(f'Out-of-bag R-2 score estimate: {rf.oob_score_:>5.3}')
    print(f'Test data R-2 score: {test_score:>5.3}')
    
       
    #==========================================================================
    #Model residuals 
    #==========================================================================
    
    #y_test - y_pred
     
    R2 = test_score
    
    return R2

def RandomForestRegression_xy(features,N_train, N_test,name):
    #Create random forest regression model
    #input: Features [includes dependent and predictors vars where first column 
    #is dependent ] format:dataframe
    #N - number of points in training model (as a proportion)
    #name: 
    #Tune = 0 (no) or 1 (yes)
        
    #output: R2, Var Weights of Importance 

    #save feature names 
    feature_list = list(features.columns)
    dependent_name = feature_list[0]
    feature_list=feature_list[1:len(feature_list)]
    
    #convert to array 
    features=np.array(features)
    print(len(features))
    print(len(features)*0.5)
    #Remove missing from each row 
    features=features[~np.isnan(features).any(axis=1)]
    
    
    
    #>>>>>>>>>> Comment out if not normalizing numbers
    #Normalize variables between 0 and 1 !!
    #Coordinates of Prediction 
    
    
    
    #idlon=feature_list.index('X')
    #idlat=feature_list.index('Y')

    #features[::,0:idlon-1] = preprocessing.normalize(features[::,0:idlon-1])
    
    
    
    
    #>>>>>>>>>>
    
  
    
     
    #Get predictors and dependent var
    y=features[:,0]
    X=features[:,1:len(features)]
    
    
    #Split into test and training set - random selection 
    #Test size is proportion saved for test 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = N_test, train_size = N_train)
    #print(len(X_train))
    
    
    
    #>>>>>>>>>> Comment here to include lat lon
    
    #Coordinates of Prediction 
    
    idlon=feature_list.index('X')
    idlat=feature_list.index('Y')

   
    ypredlon=X_test#[:,idlon]
    ypredlat=X_test#[:,idlat]
    
    X_test=X_test#[:,0:idlon]
    X_train=X_train#[:,0:idlon]
    
    #>>>>>>>>>> 
    
    
    
    # Fitting Random Forest Regression to the dataset
    #n_estimator = # of decision trees
    #rndom state 
    rng = np.random.RandomState(0)
    
    rf = RandomForestRegressor(n_estimators = 50, random_state = rng,oob_score=True)
    rf.fit(X_train, y_train)                #Run RF model using the training data 
    y_pred = rf.predict(X_test)             #Generate prediction with Test Data 
     
    test_accuracy=rf.score(X_train, y_train)
    train_accuracy =rf.score(X_test, y_test)
    print (f'Accuracy test data: {test_accuracy:>5.3}')#Test accuracy of training data
    print (f'Accuracy train data: {train_accuracy:>5.3}')#test accuracy of test data 
    #If the training accuracy is so high and the testing is not as close, it’s safe to say model is overfit on training...
    
    #MSE 
    mse=mean_squared_error(y_test, y_pred)
    #print(mse)
    #print(np.var(y_test))
    
    
    #Get Random Forest R2 for test set
    test_score = r2_score(y_test, y_pred)
    print(f'Out-of-bag R-2 score estimate: {rf.oob_score_:>5.3}')
    print(f'Test data R-2 score: {test_score:>5.3}')
    
       
    #==========================================================================
    #Model residuals 
    #==========================================================================
    
    #y_test - y_pred
    
   
    #==========================================================================
    #Feature Importance from Permutation
    #==========================================================================
    print('Generating feature importnace from permutation...')
    # #Get list of PERMUTATION feature importances 
    result=permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=1)
    sorted_idx = result.importances_mean.argsort()
    Xlabel = pd.DataFrame(columns=feature_list)
 
        
    fig, ax = plt.subplots()
    ax.boxplot(
    result.importances[sorted_idx].T, vert=False, labels=Xlabel.columns[sorted_idx])
    ax.set_xlabel("Permutation Importances")
    ax.set_title(name)
    ax.set_xlim(0,1)
    font = {'family' : 'sans',
        'weight' : 'bold',
        'size'   : 12}

    plt.rc('font', **font)
    fig.tight_layout()
    plt.show()
    
    
    
    #==========================================================================

    #==========================================================================
    #Feature Importance from Impurity Index
    #==========================================================================
   
    #get list of feature importances 
    
    
    importances = list(rf.feature_importances_)
    feature_importances=[(feature, round(importance,2)) for feature,importance in zip(feature_list,importances)]
     #sort the feature importances 
    feature_importances=sorted(feature_importances,key=lambda x: x[1], reverse =True)
     #print most important features from most to least ... 
    print('Generating feature importnace from impurity')
    [print('Variable: {:20} IMportance: {}'.format(*pair)) for pair in feature_importances];
    
    
    
    
    #==========================================================================
    #            #remove outlier
    #==========================================================================
   
    # #remove outlier
    # y_test=np.where(y_test<-np.std(y_test)*3,np.nan,y_test) #Replace very low vals with nan. 
    # y_pred=np.where(y_pred<-np.std(y_pred)*3,np.nan,y_pred)
    
    
    
    # #Find infinite Vals. 
    # finiteYmask = np.isfinite(y_pred)
    
    # Yclean = y_pred[finiteYmask]
    # Xclean = y_test[finiteYmask]
    
    # finiteXmask = np.isfinite(y_test)
    
    # Yclean = y_pred[finiteXmask]
    # Xclean = y_test[finiteXmask]
    
    # y_test=Xclean
    # y_pred=Yclean
    
  
    # ypredlon=ypredlon[finiteYmask]
    # ypredlon=ypredlon[finiteXmask]
    
    # ypredlat=ypredlat[finiteYmask]
    # ypredlat=ypredlat[finiteXmask]
    
    #regression part
    
    
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_test,y_pred)
    line = slope*y_test+intercept
  
    #Figure
    
    
    
    from matplotlib.colors import LinearSegmentedColormap

    # "Viridis-like" colormap with white background
    white_viridis = LinearSegmentedColormap.from_list('white_viridis', [
        (0, '#ffffff'),
        (1e-20, '#440053'),
        (0.2, '#404388'),
        (0.4, '#2a788e'),
        (0.6, '#21a784'),
        (0.8, '#78d151'),
        (1, '#fde624'),
        ], N=256)
    
    fig = plt.figure()
   
    ax = fig.add_subplot(1, 1, 1, projection ='scatter_density')    
    ax.grid(False)
    ax.patch.set_edgecolor('black')  
    ax.patch.set_linewidth(4) 
    density = ax.scatter_density(y_test, y_pred, vmax = 200, cmap = white_viridis)
    fig.colorbar(density, label='Number of points per pixel')
    

    
     # # #Visualizations 
    plt.plot(y_test, line, 'r', label='y={:.2f}x+{:.2f}'.format(slope,intercept))
    plt.plot(y_test, y_test, 'k:',label='1:1')
  
    #Scatter density
    # Calculate the point density
    #xy = np.vstack([y_test,y_pred]) 
    #z = gaussian_kde(xy)(xy)
  
    #plt.scatter(y_test,y_pred)#Scatter Density 
    
       
    
    plt.xlabel('observed')
    plt.ylabel('predicted')
    plt.title(dependent_name)
    
    #plt.xlim(-700,1100)
    #plt.ylim(-700,1100)
    plt.annotate("r-squared = {:.2f}".format(r2_score(y_test,y_pred)), (np.max(line)*0.25, np.min(line)))
   
    plt.legend(fontsize=14)
    plt.show()
    #End 

    
    return test_score, mse,feature_importances, feature_list,X_train, X_test, y_train, y_test,y_pred,ypredlon,ypredlat,rf
   