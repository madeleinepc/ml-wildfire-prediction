#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 13:51:16 2022

EXTRACT SHAPEFILE

@author: madeleip
"""

import os
import fiona 


def getShape(path_to_shape):
    
    files =os.listdir(path_to_shape)
    
    for f in files:
        if f.endswith('.shp'):
              fsrtm= path_to_shape +'/' + f
    
#LOAD SHAPEFILE
    with fiona.open(fsrtm, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]


    return shapes