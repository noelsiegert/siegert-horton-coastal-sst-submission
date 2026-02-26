#!/usr/bin/env python
# coding: utf-8

# ---
# ## This script follows on from prep_daily_Era5HeatFlux.py, it simply concatenates intermediate and saves. 
# #### Noel Siegert, 7/30/25- 
# ---


# (edited, re-ran this on 8/5/25 as a .py script)

# ENVIRONMENT: pangeo23

# imports
import os
import xarray as xr
import numpy as np
import netCDF4 
import glob
import pandas as pd
import geopandas as gpd
from datetime import datetime
from scipy import stats


# scriptname
script = 'coastal_sst/code/dataprep/compile_daily_Era5HeatFlux.py'

# data dir
data_dir = '/dx02/data/nsiegert/Era5HeatFluxes_proc/'
os.chdir(data_dir)


# dataframe with the stations we are using
df = pd.read_csv('/home/nsiegert/projects/coastal_sst/data/hadisd_stations_using_Expanded.csv')
df = df.drop(['Unnamed: 0'], axis=1)

# convert df into geodataframe for ease of plotting
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=df.LON, y=df.LAT))


# for each of the era5 variables:
for varname in ['surface_latent_heat_flux_dailymean', 'surface_sensible_heat_flux_dailymean', 'surface_net_solar_radiation_dailymean', 'surface_net_thermal_radiation_dailymean']:
        
    if varname=='surface_net_solar_radiation_dailymean':
        var = 'ssr'
    elif varname=='surface_net_thermal_radiation_dailymean':
        var = 'str'
    elif varname=='surface_latent_heat_flux_dailymean':
        var = 'slhf'
    elif varname=='surface_sensible_heat_flux_dailymean':
        var = 'sshf'

    # list files
    ocean_ds_list = glob.glob('1.5deg.{}.*.nc'.format(varname))
    land_ds_list =  glob.glob('OverStations.{}.*.nc'.format(varname))

    # open/compile files
    ocean_ds = xr.open_mfdataset(ocean_ds_list)
    land_ds = xr.open_mfdataset(land_ds_list)
    
    # create climatologies and anomalies (takes about 8 min per variable)
    ocean_clim = ocean_ds[var].groupby('time.dayofyear').mean(dim='time')
    land_clim = land_ds[var].groupby('time.dayofyear').mean(dim='time')
    
    ocean_anom = ocean_ds[var].groupby('time.dayofyear') - ocean_clim
    land_anom = land_ds[var].groupby('time.dayofyear') - land_clim
    
    ocean_anom_ds = ocean_anom.to_dataset()
    land_anom_ds = land_anom.to_dataset()
    
    # save out the concatenated version
    now = datetime.now()
    ocean_ds.attrs['script'] = script
    ocean_anom_ds.attrs['script'] = script
    land_ds.attrs['script'] = script
    land_anom_ds.attrs['script'] = script
    
    ocean_ds.attrs['timestamp'] = now.strftime("%Y-%m-%d %H:%M:%S")
    ocean_anom_ds.attrs['timestamp'] = now.strftime("%Y-%m-%d %H:%M:%S")
    land_ds.attrs['timestamp'] = now.strftime("%Y-%m-%d %H:%M:%S")
    land_anom_ds.attrs['timestamp'] = now.strftime("%Y-%m-%d %H:%M:%S")
    
    ocean_ds.to_netcdf('ALLSTATIONS.1.5deg.daily.{}.7.30.2025.nc'.format(var))
    ocean_anom_ds.to_netcdf('ALLSTATIONS.1.5deg.daily.{}_anom.7.30.2025.nc'.format(var))
    land_ds.to_netcdf('ALLSTATIONS.OverStations.daily.{}.7.30.2025.nc'.format(var))
    land_anom_ds.to_netcdf('ALLSTATIONS.OverStations.daily.{}_anom.7.30.2025.nc'.format(var))

    print('saved: ALLSTATIONS.1.5deg.daily.{}.7.30.2025.nc'.format(var))
    print('saved: ALLSTATIONS.OverStations.daily.{}.7.30.2025.nc'.format(var))
    print('saved: ALLSTATIONS.1.5deg.daily.{}_anom.7.30.2025.nc'.format(var))
    print('saved: ALLSTATIONS.OverStations.daily.{}_anom.7.30.2025.nc'.format(var))