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

def downloadNDVI(Year):
     """ Download VIIRS ndvi from  https://www.ncei.noaa.gov/thredds/dodsC/cdr/ndvi/2025/VIIRS-Land_v001_JP113C1_NOAA-20_20250104_c20250106153011.nc"""
     

     #URL to download
     URL = 'https://www.ncei.noaa.gov/thredds/dodsC/cdr/ndvi/' + str(Year) + '/VIIRS-Land_v001_JP113C1_NOAA-20_20250104_c20250106153011.nc'
     f = xarray.open_dataset(URL, engine = 'netcdf4')
     print('downloading from server...')
    
    
     return  f
        



def NDVI_cropped(f,var_name,path_to_shape):
    """Get cropped variable from VIIRS NDVI"""

    xds = f[var_name]
        
    xds.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
    xds.rio.write_crs("EPSG:4326", inplace=True)
    
    geodf = geopandas.read_file(path_to_shape)
    clipped = xds.rio.clip(geodf.geometry, geodf.crs)
    
    #GET VPD For DAY FIRE START          
    array = clipped
    longs = clipped['longitude']
    lats = clipped['latitude']
    
    [lats, longs] = np.meshgrid(np.array(lats), np.array(longs))
    
    return array,lats,longs 



    
