#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 11:15:25 2023

@author: madeleip
"""

import os
import rasterio 
from rasterio.plot import show
from pyproj import Proj, transform
import numpy as np
from affine import Affine
import fiona
import rasterio.mask
import rasterio
from matplotlib import pyplot
import rioxarray
import geopandas
import xarray
import subprocess
import datetime
import requests

def downloadGridMet(Year, Month, Day,variable):
     """ Download via URL from gridMET https://www.climatologylab.org/gridmet.html"""
     
     if len(str(Day))<2:
        Day1 ='0'+str(Day)
     else:
        Day1=str(Day)
        
     if len(str(Month))<2:
        Month1 ='0'+str(Month)
     else:
        Month1=str(Month)
     
     datestring_f= str(Year)+'-' + Month1 + '-' + Day1 
     print(datestring_f)

     #URL to download
     URL = 'http://thredds.northwestknowledge.net:8080/thredds/dodsC/MET/' + variable + '/' + variable +'_' + str(Year) + '.nc'
     f = xarray.open_dataset(URL, engine = 'netcdf4')
     print('downloading from server...')
    
     #Sort Time Coord
     f['day'] = np.sort(f['day'].values)
     f = f.sel(day = slice(datestring_f, datestring_f))
    
     

     return  f
        



def GridMet_cropped(f,var_name,path_to_shape):
    """Get cropped variable from gridMET"""

    xds = f[var_name]
        
    xds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    xds.rio.write_crs("EPSG:4326", inplace=True)
    
    geodf = geopandas.read_file(path_to_shape)

    clipped = xds.rio.clip(geodf.geometry, geodf.crs)
    
    #GET VPD For DAY FIRE START          
    array = clipped
    longs = clipped['lon']
    lats = clipped['lat']
    
    [lats, longs] = np.meshgrid(np.array(lats), np.array(longs))
    
    return array,lats,longs 



    
