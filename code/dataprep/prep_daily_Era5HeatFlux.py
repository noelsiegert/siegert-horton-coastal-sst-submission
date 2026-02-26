#!/usr/bin/env python
# coding: utf-8

# ---
# ## This script selects/prepares ERA5 surface heat flux data for each coastal weather station. I create average heat flux over the "Coastal SST Circle," as well as grabbing the heat flux from the gridcell in the weather station falls.
# #### Noel Siegert, 7/29/25- 
# ---

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
import sys

print('----------------------------')
ds_input_arg = sys.argv[1]
varname = ds_input_arg
print('System argument (Varname): {}'.format(ds_input_arg))
print('----------------------------')

# scriptname
script = 'coastal_sst/code/dataprep/prep_daily_Era5HeatFlux.py'

# data dir
data_dir = '/dx03/data/ERA5/'
os.chdir(data_dir)

# dataframe with the stations we are using
df = pd.read_csv('/home/nsiegert/projects/coastal_sst/data/hadisd_stations_using_Expanded.csv')
df = df.drop(['Unnamed: 0'], axis=1)

# convert df into geodataframe for ease of plotting
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=df.LON, y=df.LAT))

# SST RADIUS AROUND Station
r = 1.5

# output file
outfile = open('/home/nsiegert/projects/coastal_sst/code/dataprep/prep_{}_progress.txt'.format(varname), 'w')

if varname=='surface_net_solar_radiation_dailymean':
    var = 'ssr'
elif varname=='surface_net_thermal_radiation_dailymean':
    var = 'str'
elif varname=='surface_latent_heat_flux_dailymean':
    var = 'slhf'
elif varname=='surface_sensible_heat_flux_dailymean':
    var = 'sshf'
    
#################
## Pre-looping ##
#################

# Open the ERA5 land-sea mask -- will say sea is anything < 25% land. 
lsm = xr.open_dataset('/dx01/data/ERA5/time_invariant/land_sea_mask.nc')
lsm = lsm.rename({'latitude':'lat', 'longitude':'lon'}).sortby('lat').sel(lat=slice(-57, 70), time='2023-01-01')

# need to save the ocean mask in 2 different lon formats: one where lon goes from 0-360, one where it goes from -180 - 180...
ocnmask360 = (lsm.lsm < 0.25)
ocnmask180 = ocnmask360.copy()

fixlons = xr.where(lsm.lon >= 180, lsm.lon-360, lsm.lon) # (note: i use >= 180 here b/c "fiyr_da" will have a min. lon of -180 and max lon of 179.75 --> this makes it match)
ocnmask180['lon'] = fixlons
ocnmask180 = ocnmask180.sortby('lon')

stayr = 1989
endyr = 1994
loop = 0

while endyr<=2025: 
    
    print('---------------')
    print(stayr, endyr)
    print(np.array(range(stayr, endyr)))
    print('')
    

    ###########################################
    ## load heat flux data for this 5y chunk ##
    ###########################################

    now1 = datetime.now()
    print('loading {} data. time = {}'.format(varname, now1))
    outfile.write('\n {} - {} \n'.format(stayr, endyr))
    outfile.write('loading {} data. time = {}\n'.format(varname, now1))


    # first, gen list of <var> files to open for this 5yr period. 
    fiveyrfile_list = []
    i = 0
    for yy in np.array(range(stayr, endyr)):
        print(str(yy))
        yr_files = glob.glob('{}/{}_{}*.nc'.format(varname, varname, str(yy)))
        if i==0:
            fiveyrfile_list = yr_files
        else:
            fiveyrfile_list.extend(yr_files)
        i+=1

    # open the files for that year, just -57 thru 70
    fiyr_ds = xr.open_mfdataset(fiveyrfile_list)
    fiyr_ds = fiyr_ds.rename({'valid_time':'time', 'latitude':'lat', 'longitude':'lon'}).sortby('lat').sel(lat=slice(-57, 70))

    # select desired data variable
    fiyr_da = fiyr_ds[var]

    # save attributes for this as well
    fiyrda_attrs = fiyr_da.attrs

    if loop==0:

        # gen mask for great lakes and caspian sea
        GL_latmask = (fiyr_da.lat >= 41) * (fiyr_da.lat <= 49.5)
        GL_lonmask = (fiyr_da.lon >= (360-92)) * (fiyr_da.lon <= (360-75))

        CS_latmask = (fiyr_da.lat >= 35.5) * (fiyr_da.lat <= 48)
        CS_lonmask = (fiyr_da.lon >= 45) * (fiyr_da.lon <= 56)

        lake_ocn_mask = np.logical_not(GL_latmask * GL_lonmask) * np.logical_not(CS_latmask * CS_lonmask)

        # also generate alternate (0-360˚) lon's
        newlons360 = xr.where(fiyr_da.lon>0, fiyr_da.lon, 360+fiyr_da.lon)

    # mask out Lakes & Caspian
    fiyr_da = fiyr_da.where(lake_ocn_mask)

    ## -- maybe I need to filter out missing values? Seems like there might not be any, though? ##
    
    ## COMPUTE the 5 years worth of <Var> data - takes like 1.5 minute
    now2 = datetime.now()
    print('computing the 5 years var data. time = {}'.format(now2))
    outfile.write('computing the 5 years <var> data. time = {}\n'.format(now2))

    #fiyr_dat = fiyr_da.compute() 
    fiyr_dat = fiyr_da.compute() 
    fiyr_dat_JustOceans = fiyr_da.where(ocnmask180).compute() 

    # also save an alternate dataArray with lons going from 0-360
    fiyr_dat_JustOceans_360lons = fiyr_dat_JustOceans.copy()
    fiyr_dat_JustOceans_360lons['lon'] = newlons360
    fiyr_dat_JustOceans_360lons = fiyr_dat_JustOceans_360lons.sortby('lon')

    now3 = datetime.now()
    print('done computing var data. time = {}'.format(now3))
    outfile.write('done computing <var> data. time = {}\n'.format(now3))


    ########################################
    ## prepare each station's local <var> ##
    ## over both ocean and over land.     ##
    ########################################

    now4 = datetime.now()
    print('selecting <var> for each station. time = {}'.format(now4))
    outfile.write('selecting <var> for each station. time = {}\n'.format(now4))

    # 
    datelist = fiyr_ds.time.values # date list

    # empty array to hold each station's <var> timeseries for this 5-year chunk. shape = [station ID, time]
    sta_5yrVar_Oceanarr = np.zeros(shape=(len(gdf), len(datelist))) * np.nan # to hold var's over the "1.5˚ sst circle" field
    sta_5yrVar_Landarr = np.zeros(shape=(len(gdf), len(datelist))) * np.nan # to hold var's over the "land / station" 

    # for each station:
    stationcounter = 0

    for station in df.iterrows():

        # station info
        stalat = station[1]['LAT']
        stalon = station[1]['LON']

        ## select <var> in the closest gridcell over the station. ##
        overstation_var_5yr = fiyr_dat.sel(lat=stalat, lon=stalon, method='nearest').data


        ## Generate the "SST-Circle-Average" <var> ##

        # if the station is located within 1.5 degrees of the antimeridan, there will be issues selecting the val's
        # using the current longitude coordinate system [-180 to 180]
        if np.abs(stalon) >= 178.5: # 7/30 tested this with a few plots, only 5 cases... seems like it works...

            print('Antimeridan Lon Flag')
            # flip the station's lon from [-180, 180] coords to [0, 360]
            if stalon < 0:
                stalon = 360 + stalon

            # create a mask for all cells within an r˚ radius of the station location (using the alternate 0-360˚ lons data)
            sell = fiyr_dat_JustOceans_360lons.sel(lat=slice(stalat-r, stalat+r), lon=slice(stalon-r, stalon+r)).compute() # r x r square around station

            # ## FFOR TESTING: plot the selection
            # fig, ax = plt.subplots(subplot_kw={'projection':ccrs.PlateCarree()})
            # sell[0].plot(ax=ax, transform=ccrs.PlateCarree())
            # ax.coastlines()
            # plt.pause(0.1)
            # print('')
            
        else: # in normal conditions:
            # create a mask for all cells within an r˚ radius of the station location
            sell = fiyr_dat_JustOceans.sel(lat=slice(stalat-r, stalat+r), lon=slice(stalon-r, stalon+r)).compute() # r x r square around station

        lats = sell.lat.values
        lons = sell.lon.values
        (lon_grid, lat_grid) = np.meshgrid(lons, lats) # make into a mesh grid
        distgrid = np.sqrt(((lon_grid - stalon)**2) + ((lat_grid - stalat)**2)) # compute euclidean distance (in ˚) to station for each coord location
        selmask = distgrid <= r # turn into a mask

        # select and average all <Var? val's within that radius

        # gen lat weights
        latw2s = np.cos(np.deg2rad(sell.lat))

        # 'cut' to the shape of the 1.5˚ SST circle
        circleavg_var_5yr = sell.where(selmask).weighted(latw2s).mean(dim=['lat','lon']).data

        ## save into arrays for this 5-year chunk. ##
        sta_5yrVar_Oceanarr[stationcounter, :] = circleavg_var_5yr
        sta_5yrVar_Landarr[stationcounter, :] = overstation_var_5yr
        stationcounter += 1
        
    # back in the 5-year-chunk loop:
    
    # Save the 5-year-chunk datasets:
    # put into DataArrays, add attributes, and save
    sta_5yrVar_Ocean_da = xr.DataArray(sta_5yrVar_Oceanarr, dims=['staid', 'time'], coords={'staid':gdf.STAID, 'time':datelist}, name=var)
    sta_5yrVar_Land_da = xr.DataArray(sta_5yrVar_Landarr, dims=['staid', 'time'], coords={'staid':gdf.STAID, 'time':datelist}, name=var)

    now = datetime.now()

    for da in [sta_5yrVar_Ocean_da, sta_5yrVar_Land_da]:
        da.attrs = fiyrda_attrs
        da.attrs['script'] = script
        da.attrs['timestamp'] = now.strftime("%Y-%m-%d %H:%M:%S")

    sta_5yrVar_Ocean_da.to_dataset().to_netcdf('/dx02/data/nsiegert/Era5HeatFluxes_proc/1.5deg.{}.{}_{}.nc'.format(varname, stayr, endyr-1))
    sta_5yrVar_Land_da.to_dataset().to_netcdf('/dx02/data/nsiegert/Era5HeatFluxes_proc/OverStations.{}.{}_{}.nc'.format(varname, stayr, endyr-1))

    print('saved: /dx02/data/nsiegert/Era5HeatFluxes_proc/1.5deg.{}.{}_{}.nc'.format(varname, stayr, endyr-1))
    print('saved: /dx02/data/nsiegert/Era5HeatFluxes_proc/OverStations.{}.{}_{}.nc'.format(varname, stayr, endyr-1))
    
    now5 = datetime.now()
    print('done with these 5 years. time = {}'.format(now5))
    outfile.write('done with these 5 years. time = {}\n'.format(now5))
    
    # update year and loop counter var's
    stayr += 5
    endyr += 5
    loop += 1