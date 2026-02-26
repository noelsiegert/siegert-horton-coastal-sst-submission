#!/usr/bin/env python
# coding: utf-8

# ---
# 
# # This script selects/prepares ERA5 T (and q?) profile data for each coastal weather station. I create daily averge T profile (700 hpa and below) over the "Coastal SST Circle," as well as grabbing the same info for the gridcell in which the weather station falls.
# ### Noel Siegert, 11/10/25-
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
from scipy import stats
import sys

script = os.getcwd() + '/prep_daily_Era5TProfiles.py'

# read the input year
print('----------------------------', flush=True)
input_year = sys.argv[1]
print('System argument (year): {}'.format(input_year), flush=True)
print('----------------------------', flush=True)


# dataframe with the stations we are using
df = pd.read_csv('/glade/u/home/nsiegert/projects/coastal_sst/data/hadisd_stations_using_Expanded.csv')
df = df.drop(['Unnamed: 0'], axis=1)

# convert df into geodataframe for ease of plotting
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=df.LON, y=df.LAT))

# SST RADIUS AROUND Station
r = 1.5

# variable name
varname = 'T' # EDIT if necessary

# ouptut files will go into my scratch b/c they are intermediate files
#output_scratch_dir = '/glade/derecho/scratch/nsiegert/coastal'


#################
## Pre-looping ##
#################

# Open the ERA5 land-sea mask -- will say sea is anything < 25% land. 
lsm = xr.open_dataset('/glade/u/home/nsiegert/projects/coastal_sst/data/land_sea_mask.nc')
lsm = lsm.rename({'latitude':'lat', 'longitude':'lon'}).sortby('lat').sel(lat=slice(-57, 70), time='2023-01-01')

# need to save the ocean mask in 2 different lon formats: one where lon goes from 0-360, one where it goes from -180 - 180...
ocnmask360 = (lsm.lsm < 0.25)
ocnmask180 = ocnmask360.copy()

fixlons = xr.where(lsm.lon >= 180, lsm.lon-360, lsm.lon) # (note: i use >= 180 here b/c "fiyr_da" will have a min. lon of -180 and max lon of 179.75 --> this makes it match)
ocnmask180['lon'] = fixlons
ocnmask180 = ocnmask180.sortby('lon')

# save 180˚ longitudes and 360˚ longitudes here 
lons180 = fixlons
lons360 = lsm.lon

loop = 0

# for this given input year:
yr = str(input_year)

# for each month:
for mm in range(12):
    mo = str(mm+1).zfill(2)

    if mm==0: continue # b/c I messed up by not including all months in my original script!

    # gen all files in that month
    mo_files = glob.glob('/gdex/data/d633000/e5.oper.an.pl/YYYYMM/e5.oper.an.pl.128_130_t.ll025sc.*.nc'.replace('YYYY', yr).replace('MM', mo))
    mo_files.sort()

    # For every daily file:
    for fi in mo_files:
    #    print(fi)
        # grab day string from filename
        day = fi[-7:-5]

        print(yr, mo, day, flush=True)

        ##############################
        # open, prep ERA5 input data #
        ##############################

        # open the INPUT file for that day, just -57 thru 70, and just 700hPa to the surface
        # INPUT LONGITUDE ON THese files: 0 to 360˚
        ds = xr.open_dataset(fi).rename({'latitude':'lat', 'longitude':'lon'}).sortby('lat').sel(lat=slice(-57, 70), level=slice(700, 1000))
        ds['lon'] = lons180 # convert the input data from 0-360 to -180-180 longitude
        ds = ds.sortby('lon') 

        # caspian and great lakes masking
        if loop==0:

            # gen mask for great lakes and caspian sea
            GL_latmask = (ds.lat >= 41) * (ds.lat <= 49.5)
            GL_lonmask = (ds.lon >= (-92)) * (ds.lon <= (-75)) # 180˚ lon format

            CS_latmask = (ds.lat >= 35.5) * (ds.lat <= 48)
            CS_lonmask = (ds.lon >= 45) * (ds.lon <= 56)

            lake_ocn_mask = np.logical_not(GL_latmask * GL_lonmask) * np.logical_not(CS_latmask * CS_lonmask)

        # mask out Lakes & Caspian
        ds = ds.where(lake_ocn_mask)

        # COMPUTE the daily mean t vs. altitude (takes about 17s, 1.85gb of memory) 
        dailymean_da = ds[varname].mean(dim='time').compute()

        # select a da of just ocean data (masked out) for the sst circle selection. 
        dailymean_da_JustOceans = dailymean_da.where(ocnmask180).compute() 

        # also save an alternate dataArray with lons going from 0-360
        dailymean_da_JustOceans_360lons = dailymean_da_JustOceans.copy()
        dailymean_da_JustOceans_360lons['lon'] = lons360
        dailymean_da_JustOceans_360lons = dailymean_da_JustOceans_360lons.sortby('lon')


        ################################################
        ## prepare each station's local <var> profile ##
        ## over both ocean and over land.             ##
        ################################################

        # for each station:
        stationcounter = 0

        # empty array to hold each station's <var> profile for this 1-day chunk. shape = [station ID, p-level]
        sta_dayVar_Oceanarr = np.zeros(shape=(len(gdf), len(dailymean_da.level))) * np.nan # to hold var's over the "1.5˚ sst circle" field
        sta_dayVar_Landarr = np.zeros(shape=(len(gdf), len(dailymean_da.level))) * np.nan # to hold var's over the "land / station" 

        for station in df.iterrows():

            # station info
            stalat = station[1]['LAT']
            stalon = station[1]['LON']


            ## select <var> in the closest gridcell over the station. ##
            overstation_var_day = dailymean_da.sel(lat=stalat, lon=stalon, method='nearest').data

            ## Generate the "SST-Circle-Average" <var> ##

            # if the station is located within 1.5 degrees of the antimeridan, there will be issues selecting the val's
            # using the current longitude coordinate system [-180 to 180]
            if np.abs(stalon) >= 178.5: # 7/30 tested this with a few plots, only 5 cases... seems like it works...

#                    print('Antimeridan Lon Flag')
                # flip the station's lon from [-180, 180] coords to [0, 360]
                if stalon < 0:
                    stalon = 360 + stalon

                # create a mask for all cells within an r˚ radius of the station location (using the alternate 0-360˚ lons data)
                sell = dailymean_da_JustOceans_360lons.sel(lat=slice(stalat-r, stalat+r), lon=slice(stalon-r, stalon+r)).compute() # r x r square around station

                # ## FFOR TESTING: plot the selection
                # fig, ax = plt.subplots(subplot_kw={'projection':ccrs.PlateCarree()})
                # sell[0].plot(ax=ax, transform=ccrs.PlateCarree())
                # ax.coastlines()
                # plt.pause(0.1)
                # print('')

            else: # in normal conditions:
                # create a mask for all cells within an r˚ radius of the station location
                sell = dailymean_da_JustOceans.sel(lat=slice(stalat-r, stalat+r), lon=slice(stalon-r, stalon+r)).compute() # r x r square around station

            lats = sell.lat.values
            lons = sell.lon.values
            (lon_grid, lat_grid) = np.meshgrid(lons, lats) # make into a mesh grid
            distgrid = np.sqrt(((lon_grid - stalon)**2) + ((lat_grid - stalat)**2)) # compute euclidean distance (in ˚) to station for each coord location
            selmask = distgrid <= r # turn into a mask

            # select and average all <Var? val's within that radius

            # gen lat weights
            latw2s = np.cos(np.deg2rad(sell.lat))

            # 'cut' to the shape of the 1.5˚ SST circle
            circleavg_var_day = sell.where(selmask).weighted(latw2s).mean(dim=['lat','lon']).data

            ## save into arrays for this 5-year chunk. ##
            sta_dayVar_Oceanarr[stationcounter, :] = circleavg_var_day
            sta_dayVar_Landarr[stationcounter, :] = overstation_var_day
            stationcounter += 1


        # back in the day-chunk loop:

        # Save the day-chunk datasets:
        # put into DataArrays, add attributes, and save
        sta_dayVar_Ocean_da = xr.DataArray(sta_dayVar_Oceanarr, dims=['staid', 'level'], coords={'staid':gdf.STAID, 'level':dailymean_da.level}, name=varname)
        sta_dayVar_Land_da = xr.DataArray(sta_dayVar_Landarr, dims=['staid', 'level'], coords={'staid':gdf.STAID, 'level':dailymean_da.level}, name=varname)

        now = datetime.now()

        for da in [sta_dayVar_Ocean_da, sta_dayVar_Land_da]:
            da.attrs = ds.attrs
            da.attrs['script'] = script
            da.attrs['timestamp'] = now.strftime("%Y-%m-%d %H:%M:%S")

        sta_dayVar_Ocean_da.to_dataset().to_netcdf('/glade/derecho/scratch/nsiegert/coastal/1.5deg.{}profile.{}{}{}.nc'.format(varname, yr, mo, day))
        sta_dayVar_Land_da.to_dataset().to_netcdf('/glade/derecho/scratch/nsiegert/coastal/OverStations.{}profile.{}{}{}.nc'.format(varname, yr, mo, day))

        print('saved: /glade/derecho/scratch/nsiegert/coastal/1.5deg.{}profile.{}{}{}.nc'.format(varname, yr, mo, day), flush=True)
        print('saved: /glade/derecho/scratch/nsiegert/coastal/OverStations.{}profile.{}{}{}.nc'.format(varname, yr, mo, day), flush=True)

        now5 = datetime.now()
        print('done with this day. time = {}'.format(now5), flush=True)
        #outfile.write('done with these 5 years. time = {}\n'.format(now5))

        # update year and loop counter var's
        loop += 1


print('dishes are done!', flush=True)
