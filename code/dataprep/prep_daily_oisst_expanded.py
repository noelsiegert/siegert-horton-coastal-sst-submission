#!/usr/bin/env python
# coding: utf-8

# ---
# This script prepares daily OISSTv2.1 SST data
# for each coastal weather station.

# I am grabbing all SST cells within a 1.5˚ radius of the station, and taking the avg (weighting by lat). 

# Noel Siegert, 1/8/25 (updated 1/9/25 to run again, putting all stations in 1 file and save). 
# ---

# imports
import os
import xarray as xr
import numpy as np
import netCDF4 
import glob
import pandas as pd
from datetime import datetime
from scipy import stats


# scriptname
script = 'coastal_sst/code/dataprep/prep_daily_oisst_expanded.py'

# data dir
data_dir = '/dx01/data/OISST/raw/'
os.chdir(data_dir)

# output textfile for tracking progress
outfile = open('/home/nsiegert/projects/coastal_sst/code/dataprep/prep_daily_oisst_expanded_progress.txt', 'w')

# dataframe with the stations we are using
df = pd.read_csv('/home/nsiegert/projects/coastal_sst/data/hadisd_stations_using_Expanded.csv')
df = df.drop(['Unnamed: 0'], axis=1)

# # SST RADIUS AROUND Station
# r = 1.5


# # loop from 1989 - present in 5 year chunks

# stayr = 1989
# endyr = 1994
# loop = 0

# while endyr<=2025:
    
#     print('---------------')
#     print(stayr, endyr)
#     print(np.array(range(stayr, endyr)))
#     print('')
#     now1 = datetime.now()
#     print('loading sst data. time = {}'.format(now1))
#     outfile.write('loading sst data. time = {}\n'.format(now1))

    
#     ###################
#     ## load sst data ##
#     ###################
    
#     # first, gen list of sst files to open for this 5yr period. 
#     fiveyrfile_list = []
#     i = 0

#     for yy in np.array(range(stayr, endyr)):
#         print(str(yy))

#         yr_files = glob.glob('oisst-avhrr-v02r01.{}*.nc'.format(str(yy)))

#         if i==0:
#             fiveyrfile_list = yr_files
#         else:
#             fiveyrfile_list.extend(yr_files)

#         i+=1

#     # open the files for that year, just -57 thru 70
#     fiyr_ds = xr.open_mfdataset(fiveyrfile_list).sel(lat=slice(-57, 70))

#     # save the dataset attrs
#     dsattrs = fiyr_ds.attrs

#     # select sst data variable, also "drop" zlev
#     fiyr_da = fiyr_ds['sst'].sel(zlev=0.0).drop_vars('zlev')

#     # save attributes for this as well
#     fiyrda_attrs = fiyr_da.attrs

#     if loop==0:

#         # gen mask
#         GL_latmask = (fiyr_da.lat >= 41) * (fiyr_da.lat <= 49.5)
#         GL_lonmask = (fiyr_da.lon >= (360-92)) * (fiyr_da.lon <= (360-75))

#         CS_latmask = (fiyr_da.lat >= 35.5) * (fiyr_da.lat <= 48)
#         CS_lonmask = (fiyr_da.lon >= 45) * (fiyr_da.lon <= 56)

#         lake_ocn_mask = np.logical_not(GL_latmask * GL_lonmask) * np.logical_not(CS_latmask * CS_lonmask)

#     # mask out Lakes & Caspian
#     fiyr_da = fiyr_da.where(lake_ocn_mask)
    
#     # convert the sst data's longitudes from [0, 360] to [-180, 180] to match station data
#     original_lons = fiyr_da.lon
#     fixlons = xr.where(fiyr_da.lon > 180, fiyr_da.lon-360, fiyr_da.lon)
#     fiyr_da['lon'] = fixlons
#     fiyr_da = fiyr_da.sortby('lon')
    
#     # COMPUTE the 5 years worth of SST data
#     now2 = datetime.now()
#     print('computing the 5 years sst data. time = {}'.format(now2))
#     outfile.write('computing the 5 years sst data. time = {}\n'.format(now2))
    
#     fiyr_dat = fiyr_da.compute()
    
#     now3 = datetime.now()
#     print('done computing sst data. time = {}'.format(now3))
#     outfile.write('done computing sst data. time = {}\n'.format(now3))
    
#     # save an alternate dataArray with lons going from 0-360 (takes ~13s)
#     fiyr_dat_360lons = fiyr_dat.copy()
#     fiyr_dat_360lons['lon'] = original_lons
#     fiyr_dat_360lons = fiyr_dat_360lons.sortby('lon')
    
    
#     ######################################
#     ## prepare each station's local SST ##
#     ######################################
    
#     now4 = datetime.now()
#     print('selecting sst for each station. time = {}'.format(now4))
#     outfile.write('selecting sst for each station. time = {}\n'.format(now4))
    
    
#     # for each station:
#     stationcounter = 0
#     for station in df.iterrows():

#         ## save station metadata
#         staid = station[1]['STAID']
#         staname = station[1]['STANAME']
#         stalat = station[1]['LAT']
#         stalon = station[1]['LON']

#         # if the station is located within 1.5 degrees of the antimeridan, there will be issues selecting the sst val's
#         # using the current longitude coordinate system [-180 to 180]
#         if np.abs(stalon) >= 178.5:

#             print('Antimeridan Lon Flag')
#             # flip the station's lon from [-180, 180] coords to [0, 360]
#             if stalon < 0:
#                 stalon = 360 + stalon

#             # create a mask for all cells within an r˚ radius of the station location (using the alternate 0-360˚ lons data)
#             sell = fiyr_dat_360lons.sel(lat=slice(stalat-r, stalat+r), lon=slice(stalon-r, stalon+r)).compute() # r x r square around station

#         else: # in normal conditions:
#             # create a mask for all cells within an r˚ radius of the station location
#             sell = fiyr_dat.sel(lat=slice(stalat-r, stalat+r), lon=slice(stalon-r, stalon+r)).compute() # r x r square around station

#         lats = sell.lat.values
#         lons = sell.lon.values
#         (lon_grid, lat_grid) = np.meshgrid(lons, lats) # make into a mesh grid
#         distgrid = np.sqrt(((lon_grid - stalon)**2) + ((lat_grid - stalat)**2)) # compute euclidean distance (in ˚) to station for each coord location
#         selmask = distgrid <= r # turn into a mask

#         # select and average all SST val's within that radius

#         # gen lat weights
#         latw2s = np.cos(np.deg2rad(sell.lat))

#         # 'cut' to SST circle
#         circleavg_sst = sell.where(selmask).weighted(latw2s).mean(dim=['lat','lon'])

#         # save.
#         now = datetime.now()
#         circleavg_sst.attrs = fiyrda_attrs
#         circleavg_sst.attrs['script'] = script
#         circleavg_sst.attrs['timestamp'] = now.strftime("%Y-%m-%d %H:%M:%S")
#         circleavg_sst.attrs['nearby_station'] = [staid, staname]
#         circleavg_sst.to_dataset().to_netcdf('/dx02/data/nsiegert/oisst_station_cirleavg/{}.1.5deg.daily.sst{}_{}.nc'.format(staid, stayr, endyr-1))
     
#         stationcounter+=1
#         if (stationcounter % 100) == 0:
#             print(stationcounter)
    
#     ## ## 
    
    
    
#     now5 = datetime.now()
#     print('done with these 5 years. time = {}'.format(now5))
#     outfile.write('done with these 5 years. time = {}\n'.format(now5))
    
    
#     # update year and loop counter var's
#     stayr += 5
#     endyr += 5
#     loop += 1
    

## Now open all intermediate files and put each station's data into one matrix. ##

now6 = datetime.now()
outfile.write('done with generating coastal SST. now saving final file. time = {}\n'.format(now6))
    
# get list of station ID's
stalist = df.STAID.tolist()
    
# open ONE random station's coastal sst datset first (to get date list):
staid = '987530-99999'
stafiles = glob.glob('/dx02/data/nsiegert/oisst_station_cirleavg/{}.1.5deg.daily.sst*.nc'.format(staid))
sta_ds = xr.open_mfdataset(stafiles)
datelist = sta_ds.time.values # date list

# loop thru each station and save each station's sst timeseries in 1 matrix. 
# empty array to hold each station's SST timeseries. shape = [station ID, time]
sta_sst_arr = np.zeros(shape=(len(stalist), len(datelist))) * np.nan

for i, staid in enumerate(stalist):

    # open that station's coastal SST files, compute
    stafiles = glob.glob('/dx02/data/nsiegert/oisst_station_cirleavg/{}.1.5deg.daily.sst*.nc'.format(staid))
    sta_ds = xr.open_mfdataset(stafiles)
    stasstdat = sta_ds.sst.data.compute()
    
    # put into array at appropriate row
    sta_sst_arr[i, :] = stasstdat
    
# put into xr.Dataarray
sta_sst_da = xr.DataArray(data=sta_sst_arr, dims=['staid', 'time'], coords={'staid':stalist, 'time':datelist})

# save 
now = datetime.now()
sta_sst_da.attrs = {'long_name': 'Daily sea surface temperature',
                     'units': 'Celsius',
                     'valid_min': -300,
                     'valid_max': 4500,
                     'script': script,
                     'timestamp': now.strftime("%Y-%m-%d %H:%M:%S"),
                     'desc.':'"coastal SST" generated by taking lat-weighted avg. of all sst cells within 1.5˚ radius of weather station'}

sta_sst_da.name = 'sst'
sta_sst_da.to_dataset().to_netcdf('/dx02/data/nsiegert/oisst_station_cirleavg/ALLSTATIONS.1.5deg.daily.sst.1.9.2025.nc')

# ... and now I can delete intermediate files. 

now_end = datetime.now()
outfile.write('dishes are done. time = {}\n'.format(now_end))
outfile.close()