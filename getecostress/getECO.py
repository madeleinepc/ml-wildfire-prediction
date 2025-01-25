""" Download ECOSTRESS data based on date range and bounding box geometry 
    Madeleine Pascolini-Campbell madeleine.a.pascolini-campbell@jpl.nasa.gov
    Adapted using material from VITALS : https://github.com/nasa/VITALS/blob/main/python/02_Working_with_EMIT_Reflectance_and_ECOSTRESS_LST.ipynb
"""
# Import required libraries
import os
import earthaccess
import warnings
import pandas as pd
import geopandas as gpd
import math

from IPython.display import display
from shapely import geometry
from skimage import io
from datetime import timedelta
from shapely.geometry.polygon import orient
from matplotlib import pyplot as plt
import xarray
import requests

def getECO(path_to_shape, path_to_data, yyyy, mm, dd):

    """ Inputs: path_to_shape (string)
                path_to_data (string)
                yyyy (int)
                mm (int)
                dd (int)
                """
    #Load ECOSTRESS via API

    #1. Load Polygon Dimensions From Shape
    polygon = gpd.read_file(path_to_shape)
    # polygon.crs

    # Merge all Polygon geometries and create external boundary
    roi_poly = polygon.unary_union.envelope
    # Re-order vertices to counterclockwise
    roi_poly = orient(roi_poly, sign=1.0)

    #Create data frame containing bounding box of shape geometry
    df = pd.DataFrame({"Name":["ROI Bounding Box"]})
    bbox = gpd.GeoDataFrame({"Name":["ROI Bounding Box"], "geometry":[roi_poly]},crs="EPSG:4326")
    # bbox

    # Set ROI as list of exterior polygon vertices as coordinate pairs
    roi = list(roi_poly.exterior.coords)


    # ECOSTRESS Collection Query
    eco_collection_query = earthaccess.collection_query().keyword('ECOSTRESS L3G Tiled JET')
    eco_collection_query.fields(['ShortName','EntryTitle','Version']).get()

    # Data Collections for our search
    # concept_ids = ['C2076112011-LPCLOUD'] #Gridded
    concept_ids = ['C2076106409-LPCLOUD'] #TILED
    # Define Date Range

    if dd >= 21 and mm >= 1:
        date1 = str(yyyy) + '-' + str(mm) + '-' + str(dd - 20)
    elif dd < 21 and mm > 1:
        date1 = str(yyyy) + '-' + str(mm - 1) + '-' + str(30 - (20 - dd))
    elif dd < 21 and mm == 1:
        date1 = str(yyyy - 1) + '-' + str(12) + '-' + str(10)

    date2 = str(yyyy) + '-' + str(mm) + '-' + str(dd)
    date_range = (date1, date2)


    results = earthaccess.search_data(
        concept_id=concept_ids,
        polygon=roi,
        temporal=date_range,
        count=500
    )

    # Create Dataframe of Results Metadata
    results_df = pd.json_normalize(results)
    # Create shapely polygons for result
    geometries = [get_shapely_object(results[index]) for index in results_df.index.to_list()]
    # Convert to GeoDataframe
    gdf = gpd.GeoDataFrame(results_df, geometry=geometries, crs="EPSG:4326")
    # Remove results df, no longer needed
    del results_df
    # Add browse imagery links
    gdf['browse'] = [get_png(granule) for granule in results]
    gdf['shortname'] = [result['umm']['CollectionReference']['ShortName'] for result in results]
    # Preview GeoDataframe
    print(f'{gdf.shape[0]} granules total')

    # =================================================================================
    #                       Only Largest
    #sort by size and only keep largest # Sort the DataFrame by the 'size' column in descending order 
    df_sorted = gdf.sort_values(by='size', ascending=False) 
    # Keep only the row with the largest size 
    results_df = df_sorted.head(1)
    gdf = results_df.head(1)

    # =================================================================================
    # Preview GeoDataframe
    print(f'{gdf.shape[0]} granule total (largest)  ')


    # Create a list of columns to keep
    keep_cols = ['meta.concept-id','meta.native-id', 'umm.TemporalExtent.RangeDateTime.BeginningDateTime','umm.TemporalExtent.RangeDateTime.EndingDateTime','umm.CloudCover','umm.DataGranule.DayNightFlag','geometry','browse', 'shortname']
    # Remove unneeded columns
    gdf = gdf[gdf.columns.intersection(keep_cols)]
    gdf.head()

    # Rename some Columns
    gdf.rename(columns = {'meta.concept-id':'concept_id','meta.native-id':'granule',
                        'umm.TemporalExtent.RangeDateTime.BeginningDateTime':'start_datetime',
                        'umm.TemporalExtent.RangeDateTime.EndingDateTime':'end_datetime',
                        'umm.CloudCover':'cloud_cover',
                        'umm.DataGranule.DayNightFlag':'day_night'}, inplace=True)
    gdf.head()



    # Split into two dataframes - ECO and EMIT
    eco_gdf = gdf[gdf['granule'].str.contains('ECO')]

    eco_gdf.head()

    keep_granules = eco_gdf.index.to_list()
    keep_granules.sort()
    filtered_results = [result for i, result in enumerate(results) if i in keep_granules]
    # Retrieve URLS for Assets
    results_urls = [granule.data_links() for granule in filtered_results]


    # Authenticate using earthaccess
    earthaccess.login(persist=True)


    # # Open Text File and Read Lines
    # # Get requests https Session using Earthdata Login Info

    path = path_to_data + 'eco/'

    fs = earthaccess.get_requests_https_session()
    # Retrieve granule asset ID from URL (to maintain existing naming convention)
    for url in results_urls:
        
        url = str(url)

        # Remove the list brackets and extra characters 
        # Remove the list brackets and extra characters 
        formatted_url = url.strip("[]'\"").split("', '")[5]
    
        granule_asset_id = url.split(',')[5] 
        granule_asset_id = granule_asset_id.split('/')[6]

        print(granule_asset_id)
        # print(formatted_url)
        
        #get files in data path
        files = os.listdir(path)
        
        #download to the path
        os.chdir(path)

        print(files)
        if granule_asset_id not in files: 
            print(f"File does not exist in {path} ...downloading from server...")
            # Download the file
            response = requests.get(formatted_url)

            if response.status_code == 200:
                with open(granule_asset_id, "wb") as file:
                    file.write(response.content)
                print("File downloaded successfully!")
            else:
                print(f"Failed to retrieve data. Status code: {response.status_code}")
        else:
            print('File exists already!')      

    path_to_eco = path + granule_asset_id

    return path_to_eco



# Function to convert a bounding box for use in leaflet notation
def convert_bounds(bbox, invert_y=False):
    """
    Helper method for changing bounding box representation to leaflet notation

    ``(lon1, lat1, lon2, lat2) -> ((lat1, lon1), (lat2, lon2))``
    """
    x1, y1, x2, y2 = bbox
    if invert_y:
        y1, y2 = y2, y1
    return ((y1, x1), (y2, x2))


# Function to create shapely polygon of spatial coverage
def get_shapely_object(result:earthaccess.results.DataGranule):
    # Get Geometry Keys
    geo = result['umm']['SpatialExtent']['HorizontalSpatialDomain']['Geometry']
    keys = geo.keys()

    if 'BoundingRectangles' in keys:
        bounding_rectangle = geo['BoundingRectangles'][0]
        # Create bbox tuple
        bbox_coords = (bounding_rectangle['WestBoundingCoordinate'],bounding_rectangle['SouthBoundingCoordinate'],
                    bounding_rectangle['EastBoundingCoordinate'],bounding_rectangle['NorthBoundingCoordinate'])
        # Create shapely geometry from bbox
        shape = geometry.box(*bbox_coords, ccw=True)
    elif 'GPolygons' in keys:
        points = geo['GPolygons'][0]['Boundary']['Points']
        # Create shapely geometry from polygons
        shape = geometry.Polygon([[p['Longitude'],p['Latitude']] for p in points])
    else:
         raise ValueError('Provided result does not contain bounding boxes/polygons or is incompatible.')
    return(shape)

# Retrieve png browse image if it exists or first jpg in list of urls

def get_png(result:earthaccess.results.DataGranule):
    https_links = [link for link in result.dataviz_links() if 'https' in link]
    if len(https_links) == 1:
        browse = https_links[0]
    elif len(https_links) == 0:
        browse = 'no browse image'
        warnings.warn(f"There is no browse imagery for {result['umm']['GranuleUR']}.")
    else:
        browse = [png for png in https_links if '.png' in png][0]
    return(browse)


def getECO_cropped(eco_file, path_to_shape):
    """ eco file is path to where the .tif eco daily is saved 
        path_to_shape (str) """
    #Read in the GeoTIF from ECOSTRESS and reproject!

    eco_et_ds = xarray.open_rasterio(eco_file).squeeze('band', drop = True)
    eco_et_ds = eco_et_ds.rio.reproject('EPSG:4326')
    #shape
    shape_eco = gpd.read_file(path_to_shape)

    #Crop to shape
    eco_et_cropped = eco_et_ds.rio.clip(shape_eco.geometry.values, shape_eco.crs, all_touched=True)
    eco_et_cropped['lon'] = eco_et_cropped['x']
    eco_et_cropped['lat'] = eco_et_cropped['y']
    #visualize
    # eco_et_cropped.plot()

    ET = eco_et_cropped

    return ET