import requests 
import geopandas as gpd
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

def getDEM(path_to_shape, path_to_output):
    """ download srtm topo data via opentopography.org 
        inputs:
        path_to_shape : path to your shapefile .shp 
        path_to_output : where you want the dem stored
        """
    
    #change to output directory 
    os.chdir(path_to_output)

    # Define your OpenTopography API key 
    api_key = 'cd3caf53ae52c0d2077f736f53cd867c' 

    # Load the shapefile 
    geodf = gpd.read_file(path_to_shape) 

    # Define the URL for the SRTM data 
    url = 'https://portal.opentopography.org/API/globaldem' 

    # Define the parameters for the request 
    params = { 'demtype': 'SRTMGL1', # SRTMGL1 for 30m resolution, SRTMGL3 for 90m resolution 
            'south': geodf.total_bounds[1], 
            'north': geodf.total_bounds[3], 
            'west': geodf.total_bounds[0], 
            'east': geodf.total_bounds[2], 
            'outputFormat': 'GTiff', 
            'API_Key': api_key } 

    # Make the request 
    response = requests.get(url, params=params) 

    # Check if the request was successful 
    if response.status_code == 200: 
        
        with open('srtm_data.tif', 'wb') as f: 
            
            f.write(response.content) 
            print('Data downloaded successfully') 
        
    else: 
        print('Failed to download data:', response.status_code)

    return
        
def CroppedDEM(path_to_dem, shape):
    """ Get the cropped dem by shapefile"""

    
    files = os.listdir(path_to_dem)


    files = os.listdir(path_to_dem)


    for f in files: 
        if '.tif' in f:
            print('DEM exists')
            #open the tif using rasterios

            img = rasterio.open(f)
        
            #Plot the Raster 
            #out_image, out_transform = rasterio.mask.mask(img, shape, crop=True)
            out_image, out_transform = rasterio.mask.mask(img, shape, crop=True)
            
            out_image=np.squeeze(out_image)
            
            
            #T0 = img.transform  # upper-left pixel corner affine transform
            T0 =out_transform
            p1 = Proj(img.crs)
            #array = img.read(1) # pixel values
            array = out_image 
            
            # print(array.shape[1])
            # print(array.shape[2])
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

            #crop out missing data
            array = np.where(array < 0, np.nan, array)

            # plt.pcolor(array);plt.colorbar()

    return array, lats, longs