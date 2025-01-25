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

import cdsapi
import zipfile

def getERA5(Year, Month, Day, path_to_data):
     """ Download via API """

     
     dataset = "derived-era5-single-levels-daily-statistics"
     request = {
     "product_type": "reanalysis",
     "variable": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature"
     ],
     "year": '"' + str(Year) + '"',
     "month": ['"' + str(Month) + '"'],
     "day": ['"'+ str(Day) + '"'],
     "daily_statistic": "daily_mean",
     "time_zone": "utc+00:00",
     "frequency": "1_hourly"
    }
   
     client = cdsapi.Client()
     os.chdir(path_to_data + '/era5')
     zipname = client.retrieve(dataset, request).download()

     return  zipname
        



def GridMet_cropped(zipname, path_to_data,path_to_shape):
    """Get cropped variable from era5"""

    #Extract data to path
    with zipfile.ZipFile(path_to_data + 'era5/' + zipname, 'r') as zip_ref:
        zip_ref.extractall(path_to_data + 'era5/')

    #Get wind direction V
    v = xarray.open_dataset(path_to_data + 'era5/' + '10m_v_component_of_wind_stream-oper_daily-mean.nc')

    #get wind direction U
    u = xarray.open_dataset(path_to_data + 'era5/' + '10m_u_component_of_wind_stream-oper_daily-mean.nc')

    xds = v['u10']
        
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



    
