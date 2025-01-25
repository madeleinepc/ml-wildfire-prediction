#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 18:14:12 2024

@author: madeleip
"""
import os 
os.chdir('/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/codes')
import pandas as pd
import numpy as np

def DataGetter():

    os.chdir('/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/workspace/1_week_before')
    
    fnameB = '1_week'
    
    #os.chdir('/Users/madeleip/Documents/PROJECTS/FIRESENSE/research/data/workspace/1_day_before')
    #fnameB = '1_day'
    
    
    #------------------------------------------------------------------------------
    #------------------------------------------------------------------------------
    #>> Load Data 
    
    #>>>>>>>>>>>>>>>
    #               Load Fire 
    #               
    #>>>>>>>>>>>>>>>
    #Read PKL data 
    
    #For One WEEK BEFORE
    #>>>>>>
    
    
    
    df_johnson = pd.read_pickle("df_johnson.pkl")
    LonET_johnson = np.array(pd.read_pickle("df_lon_johnson.pkl"))
    LatET_johnson = np.array(pd.read_pickle("df_lat_johnson.pkl"))
    mask_johnson = np.array(pd.read_pickle("df_mask_johnson.pkl"))
    
    
    
    #Load Fire B cerro pelado
    df_cerro = pd.read_pickle("df_cerropelado.pkl")
    LonET_cerro = np.array(pd.read_pickle("df_lon_cerropelado.pkl"))
    LatET_cerro = np.array(pd.read_pickle("df_lat_cerropelado.pkl"))
    mask_cerro=np.array(pd.read_pickle("df_mask_cerropelado.pkl"))
    
    
    
    #Load Fire B
    #Read PKL data FOR hermits peak 
    df_hermits = pd.read_pickle("df.pkl")
    LonET_hermits = np.array(pd.read_pickle("df_lon.pkl"))
    LatET_hermits = np.array(pd.read_pickle("df_lat.pkl"))
    mask_hermits = np.array(pd.read_pickle("df_mask.pkl"))
    
    
    
    #Load Fire B
    #Read PKL data FOR hermits peak 
    df_black = pd.read_pickle("df_black.pkl")
    LonET_black = np.array(pd.read_pickle("df_lon_black.pkl"))
    LatET_black = np.array(pd.read_pickle("df_lat_black.pkl"))
    mask_black =np.array(pd.read_pickle("df_mask_black.pkl"))
    
    
    #Read PKL data FOR doagy
    df_doagy = pd.read_pickle("df_doagy.pkl")
    LonET_doagy = np.array(pd.read_pickle("df_lon_doagy.pkl"))
    LatET_doagy = np.array(pd.read_pickle("df_lat_doagy.pkl"))
    mask_doagy =np.array(pd.read_pickle("df_mask_doagy.pkl"))
    
    
    #Read for Beartrap 
    df_beartrap = pd.read_pickle("df_beartrap.pkl")
    LonET_beartrap = np.array(pd.read_pickle("df_lon_beartrap.pkl"))
    LatET_beartrap = np.array(pd.read_pickle("df_lat_beartrap.pkl"))
    mask_beartrap =np.array(pd.read_pickle("df_mask_beartrap.pkl"))
    
    
    #Read for McBride 
    df_mcbride = pd.read_pickle("df_mcbride.pkl")
    LonET_mcbride = np.array(pd.read_pickle("df_lon_mcbride.pkl"))
    LatET_mcbride = np.array(pd.read_pickle("df_lat_mcbride.pkl"))
    mask_mcbride =np.array(pd.read_pickle("df_mask_mcbride.pkl"))
    
    
    #Read for Cooks Peak 
    df_cookspeak = pd.read_pickle("df_cookspeak.pkl")
    LonET_cookspeak = np.array(pd.read_pickle("df_lon_cookspeak.pkl"))
    LatET_cookspeak = np.array(pd.read_pickle("df_lat_cookspeak.pkl"))
    mask_cookspeak =np.array(pd.read_pickle("df_mask_cookspeak.pkl"))
    
    return df_johnson, df_cerro, df_hermits, df_black, df_doagy, df_beartrap, df_mcbride, df_cookspeak
 
