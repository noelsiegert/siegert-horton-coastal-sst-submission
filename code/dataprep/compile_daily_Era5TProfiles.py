#!/usr/bin/env python
# coding: utf-8

# ---
#
# # This script compiles ERA5 T profile data
# ### Noel Siegert, 11/11/25-
#
# ---

# conda environment: enso_imp

# imports
import os
import xarray as xr
import numpy as np
import netCDF4 
import glob
import pandas as pd
import geopandas as gpd
from datetime import datetime

script = os.getcwd() + '/compile_daily_Era5TProfiles.py'

# input (intermediate) files
files = glob.glob('/glade/derecho/scratch/nsiegert/coastal/1.5deg.Tprofile.*.nc')
files.sort()

dates = pd.date_range('1990-01-01', '2023-12-31', freq='D')

if not len(dates)==len(files):
    print('uh-oh. len(dates)={}, len(files)={}'.format(len(dates), len(files)), flush=True)

# load all the daily files into one dataset
empty_arr = np.zeros(shape=(1474, 12, len(dates))) * np.nan

print('loading the data.', flush=True)
for i in range(len(dates)):

    # open daily dataset
    ds = xr.open_dataset(files[i])

    # put into np.array
    empty_arr[:, :, i] = ds.T.data

    # close
    ds.close()

    if i%1000==0:
        print(i, flush=True)


# put into DataArray
print('putting into dataArray', flush=True)
full_da = xr.DataArray(empty_arr, dims=['staid', 'level', 'time'], coords={'staid':ds.staid, 'level':ds.level, 'time':dates})


# add attrs
full_da.attrs = ds.T.attrs
now = datetime.now()
full_da.attrs['script'] = script
full_da.attrs['timestamp'] = now.strftime("%Y-%m-%d %H:%M:%S")

# save 1 (b/c want access to it sooner...)
print('saving full_da.', flush=True)
full_da.to_dataset(name='T').to_netcdf('/glade/u/home/nsiegert/projects/coastal_sst/data/ALLSTATIONS.1.5deg.Tprofile.nc')

# generate anomalies (from each day of year)
print('generating anomalies.', flush=True)
clim = full_da.groupby('time.dayofyear').mean()
full_da_anoms = full_da.groupby('time.dayofyear') - clim
full_da_anoms.attrs = full_da.attrs

# save
print('saving anoms.', flush=True)
full_da_anoms.to_dataset(name='T').to_netcdf('/glade/u/home/nsiegert/projects/coastal_sst/data/ALLSTATIONS.1.5deg.Tprofile_anoms.nc')

print('dishes are done!', flush=True)
