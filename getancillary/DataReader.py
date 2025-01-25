#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 14:14:07 2021

@author: madeleip
"""

"""
A python script for reading GEOTIFS and ECOSTRESS 
data. 

The following variables returned: 


"""

# Load libraries 
import numpy as np
import matplotlib 
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import xarray 
import xarray as xr
import datetime
from sklearn.linear_model import LinearRegression
import cartopy.crs as ccrs
import rasterio 
from rasterio.plot import show
from pyproj import Proj, transform
from affine import Affine
import os 
import sys
from osgeo import gdal_array
from osgeo import gdal, gdalconst
from scipy.interpolate import griddata 


def read_geo(data_geotiff):
    """Read geotif of fire burn severity from Sentinel-2 and return the lat lon
    grid and data and plot the data 

    PARAMETERS
    ----------
    data_geotiff : xarray
      Timeseries xarray containing variable to calc climatology for
  
       
    RETURNS
    -------
    1) Data array of fire burn severity,  lats, longs 
        
    """
    
    img = rasterio.open(data_geotiff)
    
    T0 = img.transform  # upper-left pixel corner affine transform
    p1 = Proj(img.crs)
    array = img.read(1) # pixel values
    
    # All rows and columns
    cols, rows = np.meshgrid(np.arange(array.shape[1]), np.arange(array.shape[0]))

    # Get affine transform for pixel centres
    T1 = T0 * Affine.translation(0.5, 0.5)
    # Function to convert pixel row/column index (from 0) to easting/northing at centre
    rc2en = lambda r, c: (c, r) * T1

    # All eastings and northings (there is probably a faster way to do this)
    eastings, northings = np.vectorize(rc2en, otypes=[float, float])(rows, cols)

    # Project all longitudes, latitudes
    p2 = Proj(proj='latlong',datum='WGS84')
    lats, longs = transform(p1, p2, eastings, northings)

    
    #define color map
    dnbr_cat_names = ["Enhanced Regrowth",
                  "Unburned",
                  "Low Severity",
                  "Moderate Severity",
                  "High Severity"]

    nbr_colors = ["g",
              "yellowgreen",
              "peachpuff",
              "coral",
              "maroon"]
    nbr_cmap = ListedColormap(nbr_colors)


    #print(array.shape)#dimensions of array
    
    # plt.imshow(array,
    #            cmap=nbr_cmap,
    #             vmin=0,
    #             vmax=0.66)
    # plt.colorbar()
    # rasterio.plot.show(img,title='dNBR',cmap=nbr_cmap, vmin=0, vmax=0.66)
    
    dnbr = xr.DataArray(array, dims=['lat','lon'],
                           coords ={'lat':lats[:,1],
                                    'lon':longs[0,:]})     
    
    return  dnbr

def get_tif_lat_long(filename):
    """Read geotif of fire burn severity from Sentinel-2 and return the lat lon
    grid and data and plot the data 

    PARAMETERS
    ----------
    filename of tif 
  
       
    RETURNS
    -------
    1) Data array of  lats, longs 
        
    """
    global lats, longs
    img = rasterio.open(filename)#open data tif
    
    T0 = img.transform  # upper-left pixel corner affine transform
    p1 = Proj(img.crs)
    array = img.read(1) # pixel values
    
    # All rows and columns
    cols, rows = np.meshgrid(np.arange(array.shape[1]), np.arange(array.shape[0]))

    # Get affine transform for pixel centres
    T1 = T0 * Affine.translation(0.5, 0.5)
    # Function to convert pixel row/column index (from 0) to easting/northing at centre
    rc2en = lambda r, c: (c, r) * T1

    # All eastings and northings (there is probably a faster way to do this)
    eastings, northings = np.vectorize(rc2en, otypes=[float, float])(rows, cols)

    # Project all longitudes, latitudes
    p2 = Proj(proj='latlong',datum='WGS84')
    lats, longs = transform(p1, p2, eastings, northings)
    
    return lats, longs





def read_ecostress(foldername):
    """Read all geotifs of data return the lat lon
    grid and data and plot the data 
    Takes in a folder, and goes through all data , regridding to same grid. And returning data array 
    
    PARAMETERS
    ----------
    
    input: foldername    :location of L3 ECOSTRESS containing geotif
  
    RETURNS
    -------
    1) Data 3D x-array of hydrological variable data,  lats, longs 
        dim: time 
        dim: lat 
        dim: lon 
    """
    
    print('getting ECOSTRESS data...')
    global A, filepath, outputFile, reference_filepath
    walk_dir = foldername

    for root, subdirs, files in os.walk(walk_dir):
      #  print('--\nroot = ' + root)
        list_file_path = os.path.join(root, 'my-directory-list.txt')
       # print('list_file_path = ' + list_file_path)
    
    index=0
    
    files = np.sort(files)
    
    
    for file in files:#recursive open all files in folder 
        
        if file.endswith('.tif'):
            
            print(file)
            print(index)

            #Get the doy timestring 
            id = str.find(file,'doy')
            
            time_data = file[id+3:id+16] #this is the time in year hour min  s
            
            print(time_data)
     
            if file.endswith(".tif"):
                #Open ECOSTRESS Geotif File 
                filepath = foldername + '/' + file
                
                #print(file)
                f = rasterio.open(filepath)#open tif image
                A = f.read(1) # pixel values
                
                #Count all elements in scence #This is the total number of points 
                numcells = np.count_nonzero(A)
                
                #Count all non nan values 
                numcells_data = np.count_nonzero(~np.isnan(A))
                
                #ONLY PROCEED FOR DATA WHERE THERE IS AT LEAST 30% data
                print(numcells_data/numcells)

                if numcells_data/numcells >= 0.30:

                    if index==0:#get first list item, this is the reference file
                    
                        print(index==0)
                        #Get Lat / Lon 
                        grid_y,grid_x=get_tif_lat_long(filepath)
                        
                        A_reference = A #set reference grid
                        
                        #get reference transformation
                        referencefile = filepath#Path to reference file
                        reference = gdal.Open(referencefile, 0)
                        referenceProj = reference.GetProjection()
                        referenceTrans = reference.GetGeoTransform()
                
                        bandreference = reference.GetRasterBand(1)    
                        x = reference.RasterXSize 
                        y = reference.RasterYSize
                        
                        eco = np.zeros(grid_y.shape)*np.nan#initialize empty array of zeros to store variables wih reference grid
                    
                        yeardoy = np.nan
                        index = 0 + 1
                        
                        #Resample data to reference layer 
                        #lat,lon=get_tif_lat_long(filepath)
                            
                    if A.size != grid_y.size: #IF does not match grid
                    
                        #Get Lat / Lon 
                        grid_yy,grid_xx=get_tif_lat_long(filepath)
                        
                        #REgrid to reference grid 
                        A = griddata((grid_xx.ravel(),grid_yy.ravel()),A.ravel(),(grid_x.ravel(),grid_y.ravel()),'linear')
                        szfinal=grid_x.shape
                        
                        A =np.reshape(A,(szfinal))
                        A = np.where(A<0,np.nan,A)#replace -missing val flag with nan 
                        
                        #inputfile = foldername_cloudmask+'/'+cfile
                        
                    if A.size == grid_y.size:
                        #   print('no interp')
                        A = np.asarray(A)#convert raster to np array 
                        #remove nans 
                        A = np.where(A<0,np.nan,A)#replace -missing val flag with nan 
                        
                        #Find year-doy
                        ind=file.find('doy')
                        #YY - DAY - HH MM 
                        
                        yd = file[ind+3:ind+10]
                    
                        #YY - DAY - HH MM 
                        yd = file[ind+3:ind+14]
                        #count number of nan data points 
                        
                    eco=np.dstack((eco,A))
                    print('stacking eco vars...')
                    yeardoy =np.dstack((yeardoy,yd))
                

    #Debugging
    print(eco.shape)
    print(grid_y.shape)
    print(grid_x.shape)

    #Create a Data Array object storing the ECOSTRESS data and lon, lat info     
    var = xr.DataArray(eco, dims=['lat','lon','t'],
                            coords ={'lat':grid_y[:,1],
                                     'lon':grid_x[0,:]})   
         
    yeardoy =np.squeeze(yeardoy) #include the day of year 'doy' time stamp
    yeardoy = np.array(yeardoy)
    var["date"] = (['t'],yeardoy)#add the date as a new dim

    return var

def read_ecostress_cloud(foldername,foldername_cloudmask):
    """Read all geotifs of data return the lat lon
    grid and data and plot the data 
    Takes in a folder, and goes through all data , regridding to same grid. And returning data array 
    
    
    
    
    PARAMETERS
    ----------
    
    input: foldername    :location of L3 ECOSTRESS containing geotif
    foldername_cloudmask :Location of L2 cloud mask  
    
     
       
    RETURNS
    -------
    1) Data 3D x-array of hydrological variable data,  lats, longs 
        dim: time 
        dim: lat 
        dim: lon 
    """
    
    print('getting ECOSTRESS data...')
    global A, filepath, outputFile, reference_filepath
    walk_dir = foldername
   # print('walk_dir = ' + walk_dir)
    
    
     
    
    for root, subdirs, files in os.walk(walk_dir):
      #  print('--\nroot = ' + root)
        list_file_path = os.path.join(root, 'my-directory-list.txt')
       # print('list_file_path = ' + list_file_path)
    
    index=0
    
    files = np.sort(files)
    
    
    for file in files:#recursive open all files in folder 
        
        if file.endswith('.tif'):
            
            print(file)
            print(index)
            #----------------------------------------------------------------------
            #----------------------------------------------------------------------
    
            #get L2 cloud mask 
            
            
            cloudfiles = os.listdir(foldername_cloudmask)#Get list of cloud mask files  
            
            
            #Get the doy timestring 
            id = str.find(file,'doy')
            
            time_data = file[id+3:id+16] #this is the time in year hour min  s
            
            print(time_data)
            
            for cfile in cloudfiles:
                
                #...if contains . etc 
                if time_data in cfile:
                    print('Found matching cloud mask...')
                    
                   # print(cfile)
                    mask = rasterio.open(foldername_cloudmask+'/'+cfile, dtype='int8')
                    mask = mask.read(1)
                    mask = mask >> 3#first 5 bytes 
                    
                   #mask out missing data
                    mask = np.where(mask==-125,np.nan,mask) #mask out -999 
                    
                    #mask out values that are cloud 
                    mask = np.where(mask>0,np.nan,mask)
                    mask = np.where(mask==0,1,mask)#cloud free is 0
                    
                    print(mask.shape)
                    break
                    
                    
            #----------------------------------------------------------------------
            #----------------------------------------------------------------------
    
            
            if 'mask' in locals():
                print('mask exists')
            
                if file.endswith(".tif"):
                    
                    #Open ECOSTRESS Geotif File 
                    
                    filepath = foldername + '/' + file
                    #print(file)
                    f = rasterio.open(filepath)#open tif image
                    A = f.read(1) # pixel values
                   
                    #Count all elements in scence
                    #This is the total number of points 
                    numcells = np.count_nonzero(A)
                   
                    #Multiply by Cloud Mask to filter out bad data 
                    A = mask*A
                    
                    #Count all non nan values 
                    numcells_data = np.count_nonzero(~np.isnan(A))
                    
        
                    
                    #ONLY PROCEED FOR DATA WHERE THERE IS AT LEAST 30% data
                    
                    
                    print(numcells_data/numcells)
                   
                           
                    if index==0:#get first list item, this is the reference file
                    
                        print(index==0)
                        #Get Lat / Lon 
                        grid_y,grid_x=get_tif_lat_long(filepath)
                        
                        
                        A_reference = A
                        
                        #get reference transformation
                        
                        referencefile = filepath#Path to reference file
                        reference = gdal.Open(referencefile, 0)
                        referenceProj = reference.GetProjection()
                        referenceTrans = reference.GetGeoTransform()
                
                        bandreference = reference.GetRasterBand(1)    
                        x = reference.RasterXSize 
                        y = reference.RasterYSize
                        
                        
                        eco = np.zeros(grid_y.shape)*np.nan#empty array of zeros to store variables wih reference grid
                 #       print('First in list')
                        yeardoy= np.nan
                        
                    index=0+1
                    
                    #Resample data to reference layer 
                    #lat,lon=get_tif_lat_long(filepath)
                        
                    if A.size != grid_y.size: #IF does not match grid
                    
                       #Get Lat / Lon 
                       grid_yy,grid_xx=get_tif_lat_long(filepath)
                      
                       #REgrid to reference grid 
                       A = griddata((grid_xx.ravel(),grid_yy.ravel()),A.ravel(),(grid_x.ravel(),grid_y.ravel()),'linear')
                       szfinal=grid_x.shape
                       
                       A =np.reshape(A,(szfinal))
                    
                       # #Get input file parameters  
                       # inputfile = filepath#Path to input file
                       # input = gdal.Open(inputfile,0)
                       
                       # inputProj = input.GetProjection()
                       # inputTrans = input.GetGeoTransform()
                       
                       # #Get output file parameters 
                       # outputfile = '/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/codes/getecostress/out.tif'
                       
                       # #Path to output file
                       # driver= gdal.GetDriverByName('GTiff')
                       # output = driver.Create(outputfile, x, y, 1, bandreference.DataType)
                       # output.SetGeoTransform(referenceTrans)
                       # output.SetProjection(referenceProj)
                       
                       # gdal.ReprojectImage(input, output, inputProj, referenceProj, gdalconst.GRA_Bilinear)
                      
                        
                       # #open transformed file 
                       # f = rasterio.open(outputfile)
                       # A = f.read(1) # pixel values
                       # A=np.asarray(A)#convert raster to np array 
                       A = np.where(A<0,np.nan,A)#replace -missing val flag with nan 
                       
                       
                       #inputfile = foldername_cloudmask+'/'+cfile
                       
                       
                    if A.size == grid_y.size:
                    #   print('no interp')
                       A=np.asarray(A)#convert raster to np array 
                        #remove nans 
                       A = np.where(A<0,np.nan,A)#replace -missing val flag with nan 
                      
                    #Find year-doy
                   
                    ind=file.find('doy')
                    #YY - DAY - HH MM 
                    
                    yd = file[ind+3:ind+10]
                   
                    #YY - DAY - HH MM 
                    yd = file[ind+3:ind+14]
                    
                    #count number of nan data points 
                    
                    
                    
                    eco=np.dstack((eco,A))
                    yeardoy =np.dstack((yeardoy,yd))
                   
                del mask
            
    print('storing eco vars')   


    #Create a Data Array object storing the ECOSTRESS data and lon, lat info     
    var = xr.DataArray(eco, dims=['lat','lon','t'],
                            coords ={'lat':grid_y[:,1],
                                     'lon':grid_x[0,:]})   
    
             
    yeardoy =np.squeeze(yeardoy) #include the day of year 'doy' time stamp
    yeardoy = np.array(yeardoy)
    var["date"] = (['t'],yeardoy)#add the date as a new dim
    
    return A,var,yeardoy

def remove_nan(var):
    #ingests array 
   
    var_vec = np.asarray(np.reshape(var,(len(var[0,:])*len(var[:,0]),1))) 
    var_vec[var_vec < 0] = np.nan #replace vals below 0 with nan
    var_mat = np.reshape(var_vec,(len(var[0,:]),len(var[:,0])))
    return var_mat


def read_ecostress_disalexi(foldername,foldername_cloudmask):
    """Read all geotifs of DISALEXI DATA
    data return the lat lon
    grid and data and plot the data 
    Takes in a folder, and goes through all data , regridding to same grid. And returning data array 
    
    
    
    
    PARAMETERS
    ----------
    
    input: foldername    :location of L3 ECOSTRESS containing geotif
    foldername_cloudmask :Location of L2 cloud mask  
    
     
       
    RETURNS
    -------
    1) Data 3D x-array of hydrological variable data,  lats, longs 
        dim: time 
        dim: lat 
        dim: lon 
    """
    
    print('getting ECOSTRESS...')
    global A, filepath, outputFile, reference_filepath
    walk_dir = foldername
   # print('walk_dir = ' + walk_dir)
    
    
     
    
    for root, subdirs, files in os.walk(walk_dir):
      #  print('--\nroot = ' + root)
        list_file_path = os.path.join(root, 'my-directory-list.txt')
       # print('list_file_path = ' + list_file_path)
    
    index=0
    
    
    for file in files:#recursive open all files in folder 
        
        if file.endswith('.tif'):
            
            print(file)
            print(index)
            #----------------------------------------------------------------------
            #----------------------------------------------------------------------
    
            #get L2 cloud mask 
            
            
            cloudfiles = os.listdir(foldername_cloudmask)#Get list of UNCERTAINTY
            
            
            #Get the doy timestring 
            id = str.find(file,'doy')
            
            time_data = file[id+3:id+16] #this is the time in year hour min  s
            
            print(time_data)
            
            for cfile in cloudfiles:
                
                #...if contains . etc 
                if time_data in cfile:
                    print('Found matching cloud mask...')
                   # print(cfile)
                    mask = rasterio.open(foldername_cloudmask+'/'+cfile, dtype='int8')
                    
                    mask = mask.read(1)
                    
                    
                    #plt.imshow(mask)
                    #plt.colorbar()
                    
                    #mask = mask >> 3#first 5 bytes 
                   #mask out missing 
                    mask = np.where(mask==9999,np.nan,mask) #mask out -999 
                    #mask out values that are bad quality, not equal to 0
                    #Mask should be 0 for cloud free 
                   
                    mask = np.where(mask>0,np.nan,mask)
                    mask = np.where(mask==0,1,mask)
                    mask =np.float32(mask)
            
                    
                    print(mask.shape)
                    break
                    
                    
            #----------------------------------------------------------------------
            #----------------------------------------------------------------------
    
            
            if 'mask' in locals():
                print('mask exists')
            
                if file.endswith(".tif"):
                    
                    #Open ECOSTRESS Geotif File 
                    
                    filepath = foldername + '/' + file
                    #print(file)
                    f = rasterio.open(filepath)#open tif image
                    A = f.read(1) # pixel values
                   
                    #Count all elements in scence
                    #This is the total number of points 
                    numcells = np.count_nonzero(A)
                    A=np.where(A<0,np.nan,A)
                    A=np.where(A==9999,np.nan,A)
                    A=np.float32(A)
                   
                    #Multiply by Cloud Mask to filter out bad data 
                    A = mask*A
                    
                
                   
                    
                    #Count all non nan values 
                    numcells_data = np.count_nonzero(~np.isnan(A))
                    
        
                    
                    #ONLY PROCEED FOR DATA WHERE THERE IS AT LEAST 30% data
                    
                    
                    print(numcells_data/numcells)
                   
                           
                    if index==0:#get first list item, this is the reference file
                    
                        print(index==0)
                        #Get Lat / Lon 
                        grid_y,grid_x=get_tif_lat_long(filepath)
                        
                        
                        A_reference = A
                        
                        #get reference transformation
                        
                        referencefile = filepath#Path to reference file
                        reference = gdal.Open(referencefile, 0)
                        referenceProj = reference.GetProjection()
                        referenceTrans = reference.GetGeoTransform()
                
                        bandreference = reference.GetRasterBand(1)    
                        x = reference.RasterXSize 
                        y = reference.RasterYSize
                        
                        
                        eco = np.zeros(grid_y.shape)*np.nan#empty array of zeros to store variables wih reference grid
                 #       print('First in list')
                        yeardoy= np.nan
                        
                    index=0+1
                    
                    #Resample data to reference layer 
                    #lat,lon=get_tif_lat_long(filepath)
                        
                    if A.size != grid_y.size: #IF does not match grid
                    
                       #Get Lat / Lon 
                       grid_yy,grid_xx=get_tif_lat_long(filepath)
                      
                       #REgrid to reference grid 
                       A = griddata((grid_xx.ravel(),grid_yy.ravel()),A.ravel(),(grid_x.ravel(),grid_y.ravel()),'linear')
                       szfinal=grid_x.shape
                       
                       A =np.reshape(A,(szfinal))
                    
                       # #Get input file parameters  
                       # inputfile = filepath#Path to input file
                       # input = gdal.Open(inputfile,0)
                       
                       # inputProj = input.GetProjection()
                       # inputTrans = input.GetGeoTransform()
                       
                       # #Get output file parameters 
                       # outputfile = '/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/codes/getecostress/out.tif'
                       
                       # #Path to output file
                       # driver= gdal.GetDriverByName('GTiff')
                       # output = driver.Create(outputfile, x, y, 1, bandreference.DataType)
                       # output.SetGeoTransform(referenceTrans)
                       # output.SetProjection(referenceProj)
                       
                       # gdal.ReprojectImage(input, output, inputProj, referenceProj, gdalconst.GRA_Bilinear)
                      
                        
                       # #open transformed file 
                       # f = rasterio.open(outputfile)
                       # A = f.read(1) # pixel values
                       # A=np.asarray(A)#convert raster to np array 
                       A = np.where(A<0,np.nan,A)#replace -missing val flag with nan 
                       
                       
                       #inputfile = foldername_cloudmask+'/'+cfile
                       
                       
                    if A.size == grid_y.size:
                    #   print('no interp')
                       A=np.asarray(A)#convert raster to np array 
                        #remove nans 
                       A = np.where(A<0,np.nan,A)#replace -missing val flag with nan 
                      
                    #Find year-doy
                   
                    ind=file.find('doy')
                    #YY - DAY - HH MM 
                    
                    yd = file[ind+3:ind+10]
                   
                    #YY - DAY - HH MM 
                    yd = file[ind+3:ind+14]
                    
                    #count number of nan data points 
                    
                    
                    
                    eco=np.dstack((eco,A))
                    yeardoy =np.dstack((yeardoy,yd))
                   
                del mask
            
    print('storing eco vars')        
    var = xr.DataArray(eco, dims=['lat','lon','t'],
                            coords ={'lat':grid_y[:,1],
                                     'lon':grid_x[0,:]})   
    
             
    yeardoy =np.squeeze(yeardoy)
    yeardoy = np.array(yeardoy)
    var["date"] = (['t'],yeardoy)#add the date as a new dim
    
    return A,var,yeardoy

