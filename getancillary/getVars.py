#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 08:22:02 2022

@author: madeleip


This program reads in data and runs wildfire severity prediction model 

"""

import os 
os.chdir('/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/codes')

from getancillary.GetSRTM import GetSRTM
from getancillary.getTif import getTif
from getecostress.FilterScenes import FilterScenes
from matplotlib import pyplot as plt
import fiona 
from getancillary.getShape import getShape
from getancillary.getFWI import getFWI
from getecostress.DataReader import read_ecostress
from getecostress.DataReader import read_ecostress_disalexi
from getancillary.getSMAP import getSMAP
import numpy as np 
from getancillary.getDNBR import getDNBR
from regrid.regridinputs import regridinputs, regridinputs2
from randomforest.RandomForestRegression import RandomForestRegression 
import elevation
import richdem as rd
import pandas as pd
import shapefile as shp
from getancillary.getLC import getLC
import xarray
from getancillary.getVPD import getVPD
from getancillary.getTMAX import getTMAX

def getVars(name, yyyy, mm, dd):
    """
        This loads in all the variables for the random forest program 
        It takes in name, and month and year of fire event 
        
    """

    global LonESI, LatESI, esiptjpl_mean, esiptjpl_nearest, varptjpl_mean, varptjpl_nearest

    #------------------------------------------------------------------------------
    #1. Load Path to shapefile of Perimeter fire data 
    path_to_shape = '/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/'+name+'/shapefile'
   
    #------------------------------------------------------------------------------
    #LOAD SHAPEFILE
    shape = getShape(path_to_shape)
    
    
    
    
    
    
    
    
    
    
    #------------------------------------------------------------------------------
    #2. Path to SRTM topography data 
    path_to_srtm = '/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/'+name+'/srtm'
    
    #------------------------------------------------------------------------------
    #LOAD SRTM DATA CROPPED TO PERIMETER 
    topo,lat_topo,lon_topo=GetSRTM(path_to_srtm,shape)
    
    if np.min(topo)<-1000:
        topo=np.where(topo<0,np.nan,topo)
    
   
    
    
    
    
    
    #------------------------------------------------------------------------------
    #3. Path to BURN SEVERITY
    path_to_dnbr = '/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/'+name+'/burnseverity'
    #FOR SBS
   # path_to_dnbr = '/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/'+name+'/burnseverity/sbs'
    
    #------------------------------------------------------------------------------
    #LOAD dNBR DATA CROPPED TO PERIMETER 
    
    
    dnbr,lat_dnbr,lon_dnbr=getDNBR(path_to_dnbr,shape)
    #save 
    #dnbr = np.where(dnbr < - 1, np.nan, dnbr) #cooks
    #plt.imshow(topo)
    #plt.pcolor(longs,lats,dnbr,vmin=0,vmax=1000,cmap='pink')
    
    
    #get data here
    #https://burnseverity.cr.usgs.gov/baer/baer-imagery-support-data-download
    
    
    
    #------------------------------------------------------------------------------
    #3. Path to Fire Weather 
    path_to_fwi = '/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/fwi/GPM'
    
    #------------------------------------------------------------------------------
    #LOAD FWI DATA CROPPED TO PERIMETER 
    #mm_yy is the desired month_year, before fire
    fwi,lat_fwi,lon_fwi=getFWI(path_to_fwi, path_to_shape, yyyy, mm)
    
    #------------------------------------------------------------------------------
    #4. Path to VPD
    path_to_vpd = '/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/vpd'
    
    #------------------------------------------------------------------------------
    #LOAD VPD DATA CROPPED TO PERIMETER 
    #mm_yy is the desired month_year, before fire
    vpd,lat_vpd,lon_vpd=getVPD(path_to_vpd, path_to_shape, yyyy, mm, dd)
    
   
    #------------------------------------------------------------------------------
    #4. Path to TEMP MAX
    path_to_tmax = '/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/tmax'
    
    #------------------------------------------------------------------------------
    #LOAD VPD DATA CROPPED TO PERIMETER 
    #mm_yy is the desired month_year, before fire
    tmax,lat_tmax,lon_tmax=getTMAX(path_to_tmax, path_to_shape, yyyy, mm, dd)
   
  
    
    
    
    #------------------------------------------------------------------------------
    #5. Path to soil moisture data
    #path_to_soilm = '/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/'+name+'/ecostress/soilm'
    
    #------------------------------------------------------------------------------
    #LOAD soil moisture
    #LOAD dNBR DATA CROPPED TO PERIMETER 
    #soilm,lat_soilm,lon_soilm=getTif(path_to_soilm,shape)
  
  


    #------------------------------------------------------------------------------
    #4B. Path to SMAP soil moisture data
    path_to_smap = '/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/'+name+'/smap'
    
    #------------------------------------------------------------------------------
    #LOAD soil moisture
    #LOAD dNBR DATA CROPPED TO PERIMETER 
    
    smap,lon_smap,lat_smap=getSMAP(path_to_smap,shape)
    # Long-term Mean Smap 
    #Remove weird outliers 
    
    smap = np.where(smap < 0, np.nan, smap)
   # smap = np.where(smap > 10000, np.nan, smap)
    smap = np.nanmean(smap,2)
    
    #FOR BLCK
    
  
    #------------------------------------------------------------------------------
    #Path to land cover data 
    #------------------------------------------------------------------------------
    #LOAD Land Cover 
    #MODIS Land Cover Type 2
    path_to_lc = '/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/'+name+'/landcover'
   
    lc, lon_lc, lat_lc = getLC(path_to_lc, shape)
    
    
    #------------------------------------------------------------------------------
    #5. Path to ECOSTRESS 
    path_to_cloud = '/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/'+name+'/ecostress/cloudmask'
    path_to_ecostress = '/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/'+name+'/ecostress/et'
    
    
    
    #------------------------------------------------------------------------------
    #LOAD Land Cover 
    
    #PT-JPL 
    min_percent = 0.50 #minimum percentof data per scene 
    
    #Store inputs 
    pathname = '/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/ancillary_vars/'
   
    #CHECK IF EXITSTS 
    if os.path.exists(pathname + name + 'varptjpl_mean.nc') == False:
    
        #Get ET data 
        A,var,yeardoy_ptjpl=read_ecostress(path_to_ecostress,path_to_cloud)
        et_ptjpl, yeardoy_ptjpl = FilterScenes(var,min_percent) #FILTER OUT DATA WITH LESS THAN x%
        varptjpl=var
        varptjpl_mean=varptjpl.mean(dim='t')
        
         
        #get sorted index of list
        ind=sorted(range(len(yeardoy_ptjpl)), key=lambda k: yeardoy_ptjpl[k])
        varptjpl_nearest=varptjpl[:,:,ind[len(ind)-1]]
        
        #Save ECOSTRESS to folder
        varptjpl_mean.to_netcdf(path = pathname + name + 'varptjpl_mean.nc')
        varptjpl_nearest.to_netcdf(path = pathname + name + 'varptjpl_nearest.nc')
        var.to_netcdf(path = pathname + name + 'var.nc')
    
    else:
    #Open Files, if exist    
        print('ECOSTRESS exists')
    
        var = xarray.open_dataset(pathname + name + 'var.nc')
        var = var['__xarray_dataarray_variable__']
        
        varptjpl_mean = xarray.open_dataset(pathname + name + 'varptjpl_mean.nc')
        varptjpl_mean = varptjpl_mean['__xarray_dataarray_variable__'] 
        
        varptjpl_nearest = xarray.open_dataset(pathname + name + 'varptjpl_nearest.nc')
        varptjpl_nearest = varptjpl_nearest['__xarray_dataarray_variable__'] 
        
    
    LonET=var.lon
    LatET=var.lat
    
    
    
    #------------------------------------------------------------------------------
    #6. Path to ECOSTRESS  ESI
    path_to_cloud= '/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/'+name+'/ecostress/cloudmask'
    path_to_ecostress = '/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/'+name+'/ecostress/esi'
    
    #------------------------------------------------------------------------------
    #LOAD Land Cover 
    
    #PT-JPL 
    min_percent = 0.50 #minimum percentof data per scene 
    
    #CHECK IF EXITSTS 
    if os.path.exists(pathname + name + 'esiptjpl_mean.nc') == False:
   
        A,esi,yeardoy_ptjpl=read_ecostress(path_to_ecostress,path_to_cloud)
        esi_ptjpl, yeardoy_ptjpl = FilterScenes(esi,min_percent) #FILTER OUT DATA WITH LESS THAN x%
        esiptjpl =esi_ptjpl
        
        esiptjpl_mean = esiptjpl.mean(dim='t')
        
        #get sorted index of list
        ind = sorted(range(len(yeardoy_ptjpl)), key=lambda k: yeardoy_ptjpl[k])
        
        
        esiptjpl_nearest = esiptjpl[:,:,ind[len(ind)-1]]
        
        esiptjpl_mean = esiptjpl_mean
        esiptjpl_nearest = esiptjpl_nearest
        
        #Save ESI ECOSTRESS to folder
        esiptjpl_mean.to_netcdf(path = pathname + name + 'esiptjpl_mean.nc')
        esiptjpl_nearest.to_netcdf(path = pathname + name + 'esiptjpl_nearest.nc')
        esi.to_netcdf(path = pathname + name + 'esi.nc')

    else:
    
        #Open Files, if exist    
        print('ECOSTRESS exists')
  
        esi = xarray.open_dataset(pathname + name + 'esi.nc')
        esi = esi['__xarray_dataarray_variable__']
      
        esiptjpl_mean = xarray.open_dataset(pathname + name + 'esiptjpl_mean.nc')
        esiptjpl_mean = esiptjpl_mean['__xarray_dataarray_variable__'] 
      
        esiptjpl_nearest = xarray.open_dataset(pathname + name + 'esiptjpl_nearest.nc')
        esiptjpl_nearest = esiptjpl_nearest['__xarray_dataarray_variable__'] 

    
    LonESI=np.array(esi.lon)
    LatESI=np.array(esi.lat)
    
    esiptjpl_mean = np.asarray(esiptjpl_mean)
    esiptjpl_nearest = np.asarray(esiptjpl_nearest)
    
    #------------------------------------------------------------------------------
    #disALEXI 
    
    # path_to_cloud_alexi= '/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/hermitspeak/ecostress/uncertainty_disalexi'
    # path_to_ecostress_alexi = '/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/hermitspeak/ecostress/et_disalexi'
    
    # A,var,yeardoy_disalexi=read_ecostress_disalexi(path_to_ecostress_alexi,path_to_cloud_alexi)
    # et_disalexi, yeardoy_disalexi = FilterScenes(var,min_percent)#FILTER OUT DATA WITH LESS THAN x%
    
    
    
    
    
    
    
    # #Find scenes that are same. 
    # inddoyptjpl=0
    # inddoydisalexi=0
    # for s in yeardoy_disalexi:
    #     if s in yeardoy_ptjpl:  
            
    #         ind = np.where(yeardoy_ptjpl==s)
    #         ind=np.array(ind)
    #         inddoyptjpl = np.dstack((inddoyptjpl,ind))
            
    #         ind = np.where(yeardoy_disalexi==s)
    #         ind=np.array(ind)
    #         inddoydisalexi = np.dstack((inddoydisalexi,ind))
            
    # inddoyptjpl=np.squeeze(inddoyptjpl)        
    # inddoyptjpl=(inddoyptjpl[2:len(inddoyptjpl)])
    
    # inddoydisalexi=np.squeeze(inddoydisalexi)        
    # inddoydisalexi=(inddoydisalexi[2:len(inddoydisalexi)])    
    
    
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    # >>>>>>>>>>>>>>> REGRID ALL INPUTS TO RESOLUTION /BOUNDARY OF ECOSTRESS <<<<<<<
    
    #Buffer
    buff=0.1
    
    if 'soilm' in locals():
        
        print('soilm exists')
        topo_ecostress, dnbr_ecostress,soilm_ecostress,smap,lc,fwi,vpd,tmax,esi_year,esi_nearest,LonET,LatET = regridinputs(topo,lon_topo,lat_topo,dnbr,lon_dnbr,lat_dnbr,soilm,lon_soilm,lat_soilm,smap,lon_smap,lat_smap,lon_lc,lat_lc, lc, lon_fwi, lat_fwi, fwi,lon_vpd,lat_vpd,vpd,lon_tmax,lat_tmax,tmax, esiptjpl_mean,esiptjpl_nearest,LonESI,LatESI,LonET,LatET,buff)
    
    else:
        print('no soilm')
        topo_ecostress, dnbr_ecostress,smap,lc,fwi,vpd,tmax,esi_year,esi_nearest,LonET,LatET = regridinputs2(topo,lon_topo,lat_topo,dnbr,lon_dnbr,lat_dnbr,smap,lon_smap,lat_smap,lon_lc,lat_lc, lc, lon_fwi, lat_fwi, fwi,lon_vpd,lat_vpd,vpd,lon_tmax,lat_tmax,tmax, esiptjpl_mean, esiptjpl_nearest, LonESI, LatESI, LonET, LatET,buff)
 
    
    #From Topo get Slope and Aspect 
    
    slope = rd.TerrainAttribute(rd.rdarray(topo_ecostress,no_data=0), attrib='slope_riserun')
    aspect = rd.TerrainAttribute(rd.rdarray(topo_ecostress,no_data=0), attrib='aspect')
    
    #------------------------------------------------------------------------------
    ## >>>>>>>>>>>>>>> Create Mask  <<<<<<<<<<<<<<<<<<
    #------------------------------------------------------------------------------
    
    
    mask = np.where(topo_ecostress > 0.0,1,np.nan)
    
    #Mask out datasets
    
    topo_ecostress = topo_ecostress * mask
    slope = slope * mask
    aspect = aspect * mask 
    dnbr_ecostress = dnbr_ecostress * mask 
    
    #------------------------------------------------------------------------------
    # >>>>>>>>>>>>>>> ReArrange Grids and Return Xarray <<<<<<<
    #------------------------------------------------------------------------------
    
    
    #convert to Pandas Data Frame
    
    #dNBR
    features=np.reshape(np.asarray(dnbr_ecostress),dnbr_ecostress.size)
    #ET ECOSTRESS
    features=np.dstack((features,np.reshape(np.asarray(varptjpl_mean),varptjpl_mean.size)))
    #ET nearesst
    features=np.dstack((features,np.reshape(np.asarray(varptjpl_nearest),varptjpl_nearest.size)))
    
    
    #ESI ECOSTRESS
    features=np.dstack((features,np.reshape(np.asarray(esi_year),esi_year.size)))
    #ESI nearesst
    features=np.dstack((features,np.reshape(np.asarray(esi_nearest),esi_nearest.size)))
    
    
    
    #SOILM 
    if 'soilm' in locals():
        features=np.dstack((features,np.reshape(np.asarray(soilm_ecostress),soilm_ecostress.size)))
   
    #SMAP 
    features=np.dstack((features,np.reshape(np.asarray(smap),smap.size)))
   
   
    #Topo Elevation
    features=np.dstack((features,np.reshape(np.asarray(topo_ecostress),topo_ecostress.size)))
    #Slope
    features=np.dstack((features,np.reshape(np.asarray(slope),slope.size)))
    #Aspect
    features=np.dstack((features,np.reshape(np.asarray(aspect),aspect.size)))
   
    #FWI 
    features=np.dstack((features,np.reshape(np.asarray(fwi),fwi.size)))
    #VPD
    features=np.dstack((features,np.reshape(np.asarray(vpd),vpd.size)))
    #TMAX
    features=np.dstack((features,np.reshape(np.asarray(tmax),tmax.size)))
   
    #Land Cover 
    features=np.dstack((features,np.reshape(np.asarray(lc),lc.size)))
    
    
    #lon
    features=np.dstack((features,np.reshape(np.asarray(LonET),LonET.size)))
    #lat
    features=np.dstack((features,np.reshape(np.asarray(LatET),LatET.size)))
    
    #lat
    features=np.squeeze(features)
        
        
    #df = pd.DataFrame(features, columns=["dNBR", "ET_year","ET_jan","ESI_year","ESI_jan","Soilm ECOSTRESS","Soilm SMAP","Elevation","Slope","Aspect","X","Y"])
    df = pd.DataFrame(features, columns=["dNBR",
                                         "ET_year",
                                         "ET_jan",
                                         "ESI_year",
                                         "ESI_jan",
                                         "SMAP",
                                         "Elevation",
                                         "Slope",
                                         "Aspect",
                                         "FWI",
                                         "VPD",
                                         "TMAX",
                                         "Land Cover",
                                         "X",
                                         "Y"])
    
    print(df)
    
    
    


    return df,mask,LonET,LatET











































#Plot ECOSTRESS Scenes Here 


# #PLOT
# plt.rc('image', cmap='YlGn')
# plt.pcolor(var.lon,var.lat,et_disalexi[:,:,inddoydisalexi].mean(dim='t'),vmin=0,vmax=10)
# plt.colorbar()
# plt.title('ET DISALEXI [mm/day]')
# plt.show()

# #Convert to mm/day
# # Watt /m2 = 0.0864 MJ /m2/day
# # MJ /m2/day  =0.408 mm /day 
# # Watt/m2 = 0.0864*0.408 mm/day = 0.035251199999999996 mm/day

# plt.rc('image', cmap='YlGn')
# plt.pcolor(varptjpl.lon,var.lat,et_ptjpl[:,:,inddoyptjpl].mean(dim='t'))
# plt.colorbar()
# plt.title('ET PT-JPL [W/m2]')
# plt.show()



# #Individually see ET images by day 

# for jj in range(0,len(inddoydisalexi)):
#     plt.rc('image', cmap='YlGn')
#     plt.pcolor(var.lon,var.lat,et_disalexi[:,:,inddoydisalexi[jj]],vmin=0,vmax=10)
#     plt.colorbar()
#     plt.xlabel('ET DISALEXI [mm/day]')
#     plt.title(yeardoy_disalexi[inddoydisalexi[jj]])
#     plt.show()
    
    
    
# for jj in range(0,len(inddoyptjpl)):
#     plt.rc('image', cmap='YlGn')
#     plt.pcolor(var.lon,var.lat,et_ptjpl[:,:,inddoydisalexi[jj]]*0.035251199999999996,vmin=0,vmax=10)
#     plt.colorbar()
#     plt.xlabel('ET PTJPL [mm/day]')
#     plt.title(yeardoy_ptjpl[inddoyptjpl[jj]])
#     plt.show()


# topo=np.where(topo==0,np.nan,topo)
# plt.rc('image', cmap='terrain')
# plt.pcolor(lon_topo,lat_topo,topo,vmin=500,vmax=4000)
# plt.colorbar()
# plt.title('SRTM Topography [m]')
# plt.show()






