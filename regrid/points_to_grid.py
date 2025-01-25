#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 18:06:59 2023

@author: madeleip
"""

#points to grid

import numpy as np
import scipy 

def points_to_grid(LonET_b, LatET_b, ypredlat, ypredlon, y_pred, y_test, B, rf_type):
    
    if rf_type =='classification':
        
        zi, yi, xi = np.histogram2d(ypredlat, ypredlon,  bins=(B,B), weights=y_pred)
        counts, _, _ = np.histogram2d(ypredlat, ypredlon,  bins=(B,B))
        
        zi = zi / counts
        zi = np.ma.masked_invalid(zi)
        
        
        zi[zi > 0] = 1
        zi[zi < 1] = np.nan
        
        XX,YY = np.meshgrid(xi[1:len(xi)],yi[1:len(yi)])
        
        pred = scipy.interpolate.griddata((XX.ravel(),YY.ravel()),zi.ravel(),(LonET_b,LatET_b),'nearest');
        
        pred[pred > 0] = 1
        pred[pred < 1] = np.nan
        
        #Obs Data Method 2....
        
        zi, yi, xi = np.histogram2d(ypredlat, ypredlon,  bins=(B,B), weights=y_test)
        counts, _, _ = np.histogram2d(ypredlat, ypredlon,  bins=(B,B))
        
        zi = zi / counts
        zi = np.ma.masked_invalid(zi)
        
        
        zi[zi > 0] = 1
        zi[zi < 1] = np.nan
        
        XX,YY = np.meshgrid(xi[1:len(xi)],yi[1:len(yi)])
        
        obs = scipy.interpolate.griddata((XX.ravel(),YY.ravel()),zi.ravel(),(LonET_b,LatET_b),'nearest');
        
        obs[obs > 0] = 1
        obs[obs < 1] = np.nan

    if rf_type =='classification2':

        zi, yi, xi = np.histogram2d(ypredlat, ypredlon,  bins=(B,B), weights=y_pred)
        counts, _, _ = np.histogram2d(ypredlat, ypredlon,  bins=(B,B))
        
        zi = zi / counts
        zi = np.ma.masked_invalid(zi)
        
        
       # zi[zi > 0] = 1
       # zi[zi < 1] = np.nan
        
        XX,YY = np.meshgrid(xi[1:len(xi)],yi[1:len(yi)])
        
        pred = scipy.interpolate.griddata((XX.ravel(),YY.ravel()),zi.ravel(),(LonET_b,LatET_b),'nearest');
        
        #pred[pred > 0] = 1
       # pred[pred < 1] = np.nan
        
        #Obs Data Method 2....
        
        zi, yi, xi = np.histogram2d(ypredlat, ypredlon,  bins=(B,B), weights=y_test)
        counts, _, _ = np.histogram2d(ypredlat, ypredlon,  bins=(B,B))
        
        zi = zi / counts
        zi = np.ma.masked_invalid(zi)
        
        
       # zi[zi > 0] = 1
       # zi[zi < 1] = np.nan
        
        XX,YY = np.meshgrid(xi[1:len(xi)],yi[1:len(yi)])
        
        obs = scipy.interpolate.griddata((XX.ravel(),YY.ravel()),zi.ravel(),(LonET_b,LatET_b),'nearest');
        
      #  obs[obs > 0] = 1
       # obs[obs < 1] = np.nan

    if rf_type == 'regression':
        
        zi, yi, xi = np.histogram2d(ypredlat, ypredlon, bins=(B,B), weights=y_pred)
        counts, _, _ = np.histogram2d(ypredlat, ypredlon, bins=(B,B))
        
        zi = zi / counts
        zi = np.ma.masked_invalid(zi)
        
        
        #zi[zi > 0] = 1
        #zi[zi < 1] = np.nan
        
        XX,YY = np.meshgrid(xi[1:len(xi)],yi[1:len(yi)])
        
        pred = scipy.interpolate.griddata((XX.ravel(),YY.ravel()),zi.ravel(),(LonET_b,LatET_b),'linear');
        
        #pred[pred > 0] = 1
        #pred[pred < 1] = np.nan
        
        #Obs Data Method 2....
        
        zi, yi, xi = np.histogram2d(ypredlat, ypredlon, bins=(B,B), weights=y_test)
        counts, _, _ = np.histogram2d(ypredlat, ypredlon, bins=(B,B))
        
        zi = zi / counts
        zi = np.ma.masked_invalid(zi)
        
        
        #zi[zi > 0] = 1
        #zi[zi < 1] = np.nan
        
        XX,YY = np.meshgrid(xi[1:len(xi)],yi[1:len(yi)])
        
        obs = scipy.interpolate.griddata((XX.ravel(),YY.ravel()),zi.ravel(),(LonET_b,LatET_b),'linear');
        
        #obs[obs > 0] = 1
        #obs[obs < 1] = np.nan
        obs = obs / 1000
        pred = pred / 1000 #To convert dnbr 
        
        
        
    return obs, pred