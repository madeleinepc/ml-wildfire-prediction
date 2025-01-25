#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 12:06:46 2024

@author: madeleip

References:
https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
"""




#PROJECTS/FIRESENSE/research/codes

import os 
os.chdir('/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/codes')

import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np

from DataGetter import DataGetter


from sklearn.preprocessing import StandardScaler, OneHotEncoder

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Load Data Sets
df_johnson, df_cerro, df_hermits, df_black, df_doagy, df_beartrap, df_mcbride, df_cookspeak = DataGetter()


os.chdir('/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/codes')

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#>>>>>>>>>>>>>>> Choose train and prediction fires  <<<<<<<
#>>>>>>>>>>>>>>>  Training fire data 


df = pd.concat((df_doagy, df_cerro, df_black,  df_mcbride, df_beartrap, df_cookspeak,  df_johnson)) #ALL EXCEPT HERMITS

#df = pd.concat((df_doagy, df_johnson,  df_cerro,  df_black)) # 4 Fires



# #Rename 
df.rename(columns = {'ET_jan':'ET_nearest', 'ESI_jan':'ESI_nearest'}, inplace = True)

fnameA1 = 'Doagy, Johnson, Cerro, Black'


#To Grid for All Fires
x = np.linspace(df['X'].min(), df['X'].max(), 1000) # create an array of 1000 evenly spaced values between the minimum and maximum of df['X']
y = np.linspace(df['Y'].min(), df['Y'].max(), 1000) # create an array of 1000 evenly spaced values between the minimum and maximum of df['Y']
LonET, LatET = np.meshgrid(x, y) # create a meshgrid from x and y arrays



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# >>>>>>>>>>>>>>> Other Group Fires <<<<<<<


df_b = pd.concat((df_mcbride, df_beartrap, df_cookspeak, df_hermits))
fnameA ='Mcbride, CooksPeak, Beartrap, Hermits'



#To Grid for All Fires
x = np.linspace(df_b['X'].min(), df_b['X'].max(), 1000) # create an array of 1000 evenly spaced values between the minimum and maximum of df['X']
y = np.linspace(df_b['Y'].min(), df_b['Y'].max(), 1000) # create an array of 1000 evenly spaced values between the minimum and maximum of df['Y']
LonET_b, LatET_b = np.meshgrid(x, y) # create a meshgrid from x and y arrays


#Rename 
df_b.rename(columns = {'ET_jan':'ET_nearest', 'ESI_jan':'ESI_nearest'}, inplace = True)


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# >>>>>>>>>>>>>>> For ENTIRE NOTH AREA  <<<<<<<
#Drop vars



df = df.drop(columns ={'SMAP', 'FWI','Land Cover'})
df_b=df_b.drop(columns={'SMAP', 'FWI','Land Cover'})
fnameC = 'no-SMAP-FWI'


#------------------------------------------------------------------------------
# Run through NN 
#------------------------------------------------------------------------------


#Remove all NaNs
df=df[~np.isnan(df).any(axis=1)]


#Target Variable y
y = df.pop('dNBR')


#Training Variables X
X = df


#Split into Test and Train / random split
from sklearn.model_selection import train_test_split
N_test = 0.5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = N_test, train_size = 1-N_test)


#Preserve X and Y coords
ypredlon=X_test['X']
ypredlat=X_test['Y']


X_test = X_test.drop(columns = {'X','Y'})
X_train = X_train.drop(columns = {'X','Y'})
y_test = y_test.drop(columns = {'X','Y'})
y_train = y_train.drop(columns = {'X','Y'})


input_shape = [X.shape[1]]
print("Input shape: {}".format(input_shape))


#Early stopping
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,#Keep weights with best loss
)


#BatchNormalization - normalize data so weights are normalized
#Dropout - Reduce overfitting of data set


#Model w/drop out (prevents overfitting)
model = keras.Sequential([
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu', input_shape=input_shape),#'rectified linear' relu unit used . This is function applied to neurons 
    layers.Dropout(rate=0.3), # apply 30% dropout to the next layer
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),   
    layers.Dropout(rate=0.3), # apply 30% dropout to the next layer
    layers.BatchNormalization(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(rate=0.3), # apply 30% dropout to the next layer
    layers.BatchNormalization(),
    layers.Dense(1),
])


#Compile the model
model.compile(
    optimizer='adam', #adam = general purpose optimizer
    loss='mae', #mean absolute error [the loss function]
    metrics =['accuracy']
)


#Fit the Model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=128,
    epochs=30,
    callbacks=[early_stopping], # put your callbacks in a list
    verbose=0 #turn off training logs
)




# convert the training history to a dataframe
#Stochastic Gradient Desccent
history_df = pd.DataFrame(history.history)
history_df.loc[:, ['loss', 'val_loss']].plot();
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))



#Now make predictions from test data
y_pred = model.predict(X_test)
y_pred = np.squeeze(y_pred)



#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# >>>>>>>>>>>>>>> SCATTER <<<<<<<
from matplotlib import pyplot as plt

plt.scatter(ypredlon,ypredlat, c = y_pred, s =1)
plt.colorbar()
plt.title('DNN - Predicted')
plt.show()

plt.scatter(ypredlon,ypredlat, c = y_test, s =1)
plt.colorbar()
plt.title('DNN - Observed')
plt.show()

plt.scatter(ypredlon,ypredlat, c = y_test-y_pred, s =1)
plt.colorbar()
plt.title('DNN - Observed - Predicted')
plt.show()


#------------------------------------------------------------------------------
#  SCATTER DENSITY PLOT 
#------------------------------------------------------------------------------
 


from matplotlib import pyplot as plt
plt.scatter(y_pred, y_test)

from scipy import stats
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
import mpl_scatter_density # adds projection='scatter_density'

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


#------------------------------------------------------------------------------
#                  Now make predictions from test data for OTHER REGION
#------------------------------------------------------------------------------

#Remove all NaNs
df_b=df_b[~np.isnan(df_b).any(axis=1)]

#Target Variable y
yb = df_b.pop('dNBR')


#Training Variables X
Xb = df_b


#Split into Test and Train 
from sklearn.model_selection import train_test_split
N_test = 0.5
X_trainb, X_testb, y_trainb, y_testb = train_test_split(Xb, yb, test_size = N_test, train_size = 1-N_test)


#Preserve X and Y coords
ypredlonb=X_testb['X']
ypredlatb=X_testb['Y']


X_testb = X_testb.drop(columns = {'X','Y'})
X_trainb = X_trainb.drop(columns = {'X','Y'})
y_testb = y_testb.drop(columns = {'X','Y'})
y_trainb = y_trainb.drop(columns = {'X','Y'})


#Now make predictions from test data for OTHER REGION
y_predb = model.predict(X_testb)
y_predb = np.squeeze(y_predb)


#------------------------------------------------------------------------------
# >>>>>>>>>>>>>>> SCATTER DENSITY PLOT  <<<<<<<
#------------------------------------------------------------------------------

from matplotlib import pyplot as plt
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(y_testb,y_predb)
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
import mpl_scatter_density # adds projection='scatter_density'

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1, projection ='scatter_density')    
ax.grid(False)
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth(4) 
density = ax.scatter_density(y_testb, y_predb, vmax = 200, cmap = white_viridis)
fig.colorbar(density, label='Number of points per pixel')

 # # #Visualizations 
plt.plot(y_test, line, 'r', label='y={:.2f}x+{:.2f}'.format(slope,intercept))
plt.plot(y_test, y_test, 'k:',label='1:1')

plt.scatter(ypredlonb,ypredlatb, c = y_predb, s =1)
plt.colorbar()
plt.title('DNN - Predicted')
plt.show()

plt.scatter(ypredlonb,ypredlatb, c = y_testb, s =1)
plt.colorbar()
plt.title('DNN - Observed')
plt.show()

