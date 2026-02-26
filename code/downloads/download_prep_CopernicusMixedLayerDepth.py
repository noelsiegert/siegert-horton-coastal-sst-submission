#!/usr/bin/env python
# coding: utf-8

# ---
# ## Download and prepare Copernicus Global Ocean Physics Reanalysis Mixed Layer Depth (MLD) data for each coastal station location
# #### 9/11/25
# ---


# Information about the dataset: https://documentation.marine.copernicus.eu/PUM/CMEMS-GLO-PUM-001-030.pdf
# https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030/description

# I went to the 2nd link --> Data Access > Daily > Subset > (click) Form --> check boxes for desired var... --> Automate > Python API

# kernel: copernicusmarine

# - For each location:
#     - use copernicusmarine to remotely open the MLD data in a 3˚ box around the station location
#     - Use the “circular mask” to “cut” the MLD data to the circle shape around the station
#     - Take the lat-weighted mean of MLD for that station
#     - add that to my outupt array
# - Save into a dataset of shape (station, time)

import os
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
import copernicusmarine

script = os.getcwd() + '/download_prep_CopernicusMixedLayerDepth.py'

# dataframe with the stations we are using
df = pd.read_csv('/home/nsiegert/projects/coastal_sst/data/hadisd_stations_using_Expanded.csv')
df = df.drop(['Unnamed: 0'], axis=1)

# radius, in ˚C
r = 1.5

# provide copernicus marine credentials
copernicusmarine.login()

# output array
MLD_out_arr = np.zeros(shape=(len(df), 10408)) # dims = [staid, time]

stanum = 0
for station in df.iterrows(): 

    # station info
    staid = station[1]['STAID']
    stalat = station[1]['LAT']
    stalon = station[1]['LON']
    
    if stanum%10==0:
        print(stanum, stalat, stalon)
    
    # FOR NOW set lat/lon bounds of the MLD data we'll subset
    maxlat = stalat + r
    minlat = stalat - r
    maxlon = stalon + r
    minlon = stalon - r
    
#    print('opening the data.')
    # (remotely) grab Mixed layer depth data in a box around the location
    ds = copernicusmarine.open_dataset(
          dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
          variables=["mlotst"],
          minimum_longitude=minlon,
          maximum_longitude=maxlon,
          minimum_latitude=minlat,
          maximum_latitude=maxlat,
          start_datetime="1993-01-01T00:00:00", # the ds starts in '93 and only goes til 7/'21
          end_datetime="2021-06-30T00:00:00",
          minimum_depth=0.49402499198913574,
          maximum_depth=0.49402499198913574,
        ).rename({'latitude':'lat', 'longitude':'lon'})    
        
    # compute... & sel. mixed layer depth
#    print('compute.')
    da = ds.mlotst.compute() ## do this at a different step

    # select a r (1.5˚) circle around the station location
    lats = da.lat.values
    lons = da.lon.values
    (lon_grid, lat_grid) = np.meshgrid(lons, lats) # make into a mesh grid
    distgrid = np.sqrt(((lon_grid - stalon)**2) + ((lat_grid - stalat)**2)) # compute euclidean distance (in ˚) to station for each coord location
    selmask = distgrid <= r # turn into a mask

    # gen lat weights
    latw2s = np.cos(np.deg2rad(da.lat))

    # 'cut' to the shape of the 1.5˚ SST circle
    circleavg_MLD = da.where(selmask).weighted(latw2s).mean(dim=['lat','lon']).data ## this takes next-to-no-time. 
    
    # add to the output array
    MLD_out_arr[stanum, :] = circleavg_MLD
    stanum += 1


# add attrs and save.
print('SAVING')
now = datetime.now()
MLD_out_da = xr.DataArray(MLD_out_arr, dims=['staid', 'time'], coords={'staid':df.STAID, 'time':ds.time}, attrs=da.attrs, name='mld')
MLD_out_da.attrs['script'] = script
MLD_out_da.attrs['timestamp'] = now.strftime("%Y-%m-%d %H:%M:%S")
MLD_out_da.to_dataset().to_netcdf('/dx02/data/nsiegert/coastal_mhw_data/ALLSTATIONS.mld.nc')

print('done.')