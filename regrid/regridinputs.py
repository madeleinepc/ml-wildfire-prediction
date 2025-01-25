#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 15:13:26 2022

@author: madeleip
"""


import scipy
from datetime import datetime,date 
import numpy as np 

def regridinputs(lon_ref, lat_ref, x, y, vv):
    """ match grid to some original grid 
         lon_ref = ref et
         lat_ref = ref et
         x = some lon grid to reshape
         y = some lat gird to reshape
         vv = some variable to reshape 
          """
    
    buff = 0.1
    #Set the reference grid 
    (lon_ref,lat_ref) = np.meshgrid(lon_ref, lat_ref,copy=False)
    (x, y) = np.meshgrid(x, y, copy = False)

    maxLat = np.max(lat_ref);
    minLat = np.min(lat_ref);
    maxLon = np.max(lon_ref);
    minLon = np.min(lon_ref);
    
    # Crop 
    I1 = np.argwhere((y[:,0]< np.min(np.min(lat_ref))-buff) | (y[:,0]> np.max(np.max(lat_ref))+buff)); # 0.5 degrees offset
    I2 = np.argwhere((x[0,:]< np.min(np.min(lon_ref))-buff) | (x[0,:]> np.max(np.max(lon_ref))+buff)); # 0.5 degrees offset
    
    I1=list(np.squeeze(I1))
    I2=list(np.squeeze(I2))
    
    
    
    #CROP based on indices
    #lat=np.delete(lat[:,0],I1)
    
    lat_ref = np.delete(lat_ref,I1,0)
    lat_ref = np.delete(lat_ref,I2,1)
    lat_ref = np.squeeze(lat_ref)
   
    
    lon_ref = np.delete(lon_ref,I1,0)
    lon_ref = np.delete(lon_ref,I2,1)
    lon_ref = np.squeeze(lon_ref)
   
    #SUBSET  TOPO 
    vv = np.delete(vv,I1,0)
    vv = np.delete(vv,I2,1)
       
    
    vv_interp = scipy.interpolate.griddata((x.ravel(),y.ravel()),vv.ravel(),(lon_ref,lat_ref),'nearest')
    
    
    return vv_interp


def regridinputs2(lon_ref, lat_ref, x, y, vv):
    """ match grid to some original grid 
         lon_ref = ref et
         lat_ref = ref et
         x = some lon grid to reshape
         y = some lat gird to reshape
         vv = some variable to reshape 
          """
    
    buff = 0.1
    #Set the reference grid 
    (lon_ref,lat_ref) = np.meshgrid(lon_ref, lat_ref,copy=False)
    
    print(lon_ref.shape)
    print(x.shape)
    # maxLat = np.max(lat_ref);
    # minLat = np.min(lat_ref);
    # maxLon = np.max(lon_ref);
    # minLon = np.min(lon_ref);
    
    # # Crop 
    # I1 = np.argwhere((y[:,0]< np.min(np.min(lat_ref))-buff) | (y[:,0]> np.max(np.max(lat_ref))+buff)); # 0.5 degrees offset
    # I2 = np.argwhere((x[0,:]< np.min(np.min(lon_ref))-buff) | (x[0,:]> np.max(np.max(lon_ref))+buff)); # 0.5 degrees offset
    
    # I1=list(np.squeeze(I1))
    # I2=list(np.squeeze(I2))
    
    
    
    # #CROP based on indices
    # #lat=np.delete(lat[:,0],I1)
    
    # lat_ref = np.delete(lat_ref,I1,0)
    # lat_ref = np.delete(lat_ref,I2,1)
    # lat_ref = np.squeeze(lat_ref)
   
    
    # lon_ref = np.delete(lon_ref,I1,0)
    # lon_ref = np.delete(lon_ref,I2,1)
    # lon_ref = np.squeeze(lon_ref)
   
    # #SUBSET  TOPO 
    # vv = np.delete(vv,I1,0)
    # vv = np.delete(vv,I2,1)
       
    
    vv_interp = scipy.interpolate.griddata((x.ravel(),y.ravel()),vv.ravel(),(lon_ref,lat_ref),'linear')
    
    
    return vv_interp



def regridinputs3(lon_ref, lat_ref, x, y, vv):
    """ match grid to some original grid 
         lon_ref = ref et
         lat_ref = ref et
         x = some lon grid to reshape
         y = some lat gird to reshape
         vv = some variable to reshape 
          """
    
    buff = 0.1
    #Set the reference grid 
    (lon_ref,lat_ref) = np.meshgrid(lon_ref, lat_ref,copy=False)
    (x, y) = np.meshgrid(x, y, copy = False)

    print(lon_ref.shape)
    print(x.shape)
    # maxLat = np.max(lat_ref);
    # minLat = np.min(lat_ref);
    # maxLon = np.max(lon_ref);
    # minLon = np.min(lon_ref);
    
    # # Crop 
    # I1 = np.argwhere((y[:,0]< np.min(np.min(lat_ref))-buff) | (y[:,0]> np.max(np.max(lat_ref))+buff)); # 0.5 degrees offset
    # I2 = np.argwhere((x[0,:]< np.min(np.min(lon_ref))-buff) | (x[0,:]> np.max(np.max(lon_ref))+buff)); # 0.5 degrees offset
    
    # I1=list(np.squeeze(I1))
    # I2=list(np.squeeze(I2))
    
    
    
    # #CROP based on indices
    # #lat=np.delete(lat[:,0],I1)
    
    # lat_ref = np.delete(lat_ref,I1,0)
    # lat_ref = np.delete(lat_ref,I2,1)
    # lat_ref = np.squeeze(lat_ref)
   
    
    # lon_ref = np.delete(lon_ref,I1,0)
    # lon_ref = np.delete(lon_ref,I2,1)
    # lon_ref = np.squeeze(lon_ref)
   
    # #SUBSET  TOPO 
    # vv = np.delete(vv,I1,0)
    # vv = np.delete(vv,I2,1)
       
    
    vv_interp = scipy.interpolate.griddata((x.ravel(),y.ravel()),vv.ravel(),(lon_ref,lat_ref),'linear')
    
    
    return vv_interp

