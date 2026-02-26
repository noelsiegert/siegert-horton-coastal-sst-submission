#!/usr/bin/env python
# coding: utf-8


# prepare_station_metadata number 2


# ---
# ## Prepare metadata in order to select desired HadISD stations
# #### Noel Siegert, 1/2/25 (last edited 1/6)
# ---


# --- 
# ## Steps:
# filter out any stations whose data either starts 6.6 years in (1997), or ends 6.6 years early (2016)
    # TIMING: ~ nothing

# filter out any stations who are > 30km (why not) away from the coast
    # TIMING: ~ 37 min for this

# filter out any stations who don't report >80% of the time (for tmax, at least)
    # takes like 1.5s per station
    # = ? an hour or so?
# --- 

# imports
import os
import glob
import numpy as np
import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
from math import radians, cos, sin, asin, sqrt


# --- 
# function defs
def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    
    Credit: https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points?noredirect=1&lq=1
    """
    # convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r

def dist_to_coastline(lat, lon, coast_gdf):
    """
    Approximates the shortest great-circle (haversine) distance between a lat-lon point and the nearest coastline.
    Coastlines are defined using the Natural Earth feautres "coastline" shapefile, 10m resolution. 
    
    Params:
    -------
         - lat, lon (float) the lat & lon coordinates to the point
         - coast_gdf (pd.geodataframe) pandas geodataframe of coastline lines/polygons (in lat-lon coordinate system)
         
         
    Returns:
    --------
        - dist2coast (float) distance to nearest coastline (km)
        - coastline_idx (int) index of the "closest coastline" in the coast_gdf 
        - closest_coast_pt_idx (int) index to the nearest line node (lat/lon point) within the "closest coastline" polygon's geometry
        
    6/21/24
    """
    
    
    ## first, ID the coastline shape that is nearest to our point
    
    distlist = []
    point = Point(lon, lat)

    # compute the distance from our point to each line (unitless)
    for i in range(len(coast_gdf)):

        distlist.append(point.distance(coast_gdf.iloc[i].geometry))

    mindist = np.min(distlist)
    coastline_idx = np.argmin(distlist) # index to closest coastline in the coast_gdf
    
    
    ## now, find the min great-circle distance (km) between our point and each "node" in the relevant coastline polygon

    node_distlist = []

    # loop thru each line node
    for clo, cla in coast_gdf.iloc[coastline_idx].geometry.coords:

        # compute the minimum dist using haversine
        node_distlist.append(haversine(lon1=lon, lat1=lat, lon2=clo, lat2=cla))

    # this will be the approx. minimum distance to the coastline in km
    dist2coast = np.min(node_distlist)
    closest_coast_pt_idx = int(np.argmin(node_distlist)) # this is the index to the closest node within the "closest line"'s geometry
    
    
    return dist2coast, coastline_idx, closest_coast_pt_idx
# --- 

## Load HadISD station metadata ##

print('loading HadISD', flush=True)
## metadata source: https://www.metoffice.gov.uk/hadobs/hadisd/v341_202405p/files/hadisd_station_fullinfo_v341_202401p.txt
## Dataset user guide: https://www.metoffice.gov.uk/hadobs/hadisd/hadisd_v340_2023f_product_user_guide.pdf

# load into dataframe
df = pd.read_fwf('/home/nsiegert/projects/coastal_sst/data/hadisd_station_fullinfo_v341_202401p.txt', widths=[12, 1, 30, 1, 7, 1, 8, 1, 7, 1, 10, 1, 10], header=None, usecols=[0, 2, 4, 6, 8, 10, 12])
df.columns = ['STAID', 'STANAME', 'LAT', 'LON', 'ELEV', 'START', 'END']


# --- 
## a. Filter out any stations whose start date is 1997 onwards, or whose end date is before 2017. ##

## (if either of these conditions are the case, they will automatically have <80% data reporting from 1990-present)

# make masks
startmask = pd.to_datetime(df['START']) <= '1997-01-01'
endmask = pd.to_datetime(df['END']) >= '2017-01-01'
datemask = (startmask * endmask)

# filter.
df = df.loc[datemask]

# --- 
## b. Filter out any stations who are >=50km from the nearest coastline ##

## Find each station's distance to the nearest coastline (in km) 
print('finding distances to coastline', flush=True)
print('len df = {}'.format(len(df)), flush=True)

# load shapefile of global coastlines from NaturalEarth features, 10m resolution (0verkill?)
## data source: https://www.naturalearthdata.com/downloads/10m-physical-vectors/10m-coastline/
coast_gdf = gpd.read_file('/home/nsiegert/projects/coastal_sst/data/ne_10m_coastline/ne_10m_coastline.shp')

## Find the distance to the coastline of each HadISD weather station
## (this should take ~37 min)
dist2coasts = []

for i in range(len(df)): 
    
    dd, min_shape_idx, closest_coast_pt_idx = dist_to_coastline(df.iloc[i].LAT, df.iloc[i].LON, coast_gdf)
    dist2coasts.append(dd)
    
    if i%100 == 0: print('dist2coast {}'.format(i), flush=True)

# add the distance to the coasts to the station df
df.insert(loc=7, column='DIST2COAST', value=dist2coasts)

# For now, keep all stations within 50km of the coast.
coastmask = (df.DIST2COAST < 50)
df = df.loc[coastmask]

# --- 
## c. Filter out stations with <=80% coverage since 1990.


# full date range from '90 thru 2020
dates = pd.date_range('1990-01-01', '2019-12-31', freq='D')

# put into a DataArray
dates_arr = xr.DataArray(data=np.ones(shape=len(dates)), dims={'time'}, coords={'time':dates})

# compute total num of days that fall in each month from 1990 - 2020 (ex. there are 930 January days)
ideal_no_obspermonth = dates_arr.groupby("time.month").sum()


tot_no_days = len(pd.date_range('1990-01-01', '2019-12-31', freq='D')) # EDIT ? currently these all go thru 2019-12-31.


pct_reporting_list = []
maxmissingmonth_list = []

# for each station
for j in range(len(df)):
    
    staid = df.iloc[j].STAID
    
    # find the file with desired station ID
    f = glob.glob('/dx01/data/hadISD/raw_station_data/*{}*.nc'.format(staid))

    # open dataset, selecting 1990 onwards
    try: 
        sta_ds = xr.open_dataset(f[0]).sel(time=slice('1990-01-01', '2023-12-31'))
    except IndexError:
        print('{} {} file not found! len(f)={}'.format(staid, df.iloc[j].STANAME, len(f)), flush=True)
        pct_reporting_list.append(np.nan)
        maxmissingmonth_list.append(np.nan)
        continue

    # compute the % of days post-1990 where the dataset has non-nan tmax data
    # (excluding QA'd out temperatures - the "valid min" is -1.1C)
    tmax_da = sta_ds.temperatures.where(sta_ds.temperatures>=-2).resample(time='D').max().compute()
    pct_reporting = round((np.sum(np.logical_not(np.isnan(tmax_da))).item() / tot_no_days) * 100, 2)
    
    # add to list
    pct_reporting_list.append(pct_reporting)
        
    # find how many missing days in each month of the station data
    sta_missing_days_per_mo = np.isnan(tmax_da).groupby("time.month").sum()

    # find station data's % missing for each month of the year
    sta_pctmissing_per_mo = (sta_missing_days_per_mo / ideal_no_obspermonth) * 100
        
    maxmissingmonth_list.append(np.max(sta_pctmissing_per_mo).item())
        
    if j%100 == 0: print('pctreport {}'.format(j), flush=True)

# either way it seems like it will cut down the sample by about half. 

# add metadata to the df
print('adding metadata.', flush=True)
#df.insert(loc=7, column='PCTREPORTING', value=pct_reporting_list)
#df.insert(loc=8, column='fail80', value=fail80_list)
#df.insert(loc=8, column='fail75', value=fail75_list)
df['PCTREPORTING'] = pct_reporting_list
df['max_pctmissing_month'] = maxmissingmonth_list

## pct reporting >80% filter
#pctreportmask = (df.PCTREPORTING > 80)
#df = df.loc[pctreportmask]

# --- 
## save that dataframe with the prepared station metadata ##
print('saving.', flush=True)
df.to_csv('/home/nsiegert/projects/coastal_sst/data/hadisd_station_preppedmetadata_v341_202401p_1_6_25.csv')
print('dishes are done.', flush=True)


# ## sources:

# https://gis.stackexchange.com/questions/222315/finding-nearest-point-in-other-geodataframe-using-geopandas
# https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points?noredirect=1&lq=1
# https://gis.stackexchange.com/questions/364058/finding-closest-point-to-shapefile-coastline-python