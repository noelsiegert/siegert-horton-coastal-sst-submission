#!/usr/bin/env python
# coding: utf-8

#---
## This script generates daily sea-level-pressure for ***1400+*** selected stations, and generates detrended anomalies.
## I've rewritten the original script so that I should be able to plug in a given variable without much issue, all I need to do is edit the file name
## and edit the lines of code between these #*#*# symbols. 
#### Noel Siegert, 11/20/25 - update to do the 11-day rolling detrend, plus generating std anomalies
#---

# kernel: pangeo23

#*#*#
# SET VAR NAME:
varname = 'slp'
#*#*#

# imports
import os
import xarray as xr
import numpy as np
import netCDF4 
import glob
import pandas as pd
from datetime import datetime
from scipy import stats

def reshape_1d_to_2d_331(da):
    """
    This function takes a 1-d timeseries of daily data (dims = [day]) to 2-d (dims = [year, day of year (1-366)])
    which makes it possible to remove the linear trend from each day's data. 
    
    Args:
        da (xr.DataArray) (must be 1-d, have a dim labeled "time", must be daily data)
        
    Returns
        yr_day_2d_da (xr.DataArray)
        
    3/31/24
    """

    # calculate number of years in the data, save important initial indices
    numyrs = len(np.unique(da.time.dt.year))
    yrlist = np.unique(da.time.dt.year)
    firstyr_num = yrlist[0]
    firstday_num = da.time.dt.dayofyear[0].item()

    # allocate blank array that has dims [row=year, columns=day of year (1-366)]
    yr_day_2darr = np.zeros(shape=(numyrs, 366)) * np.nan

    # counter var's
    firstday_idx = firstday_num - 1
    yr = firstyr_num
    yr_idx = 0
    dayctr = 0 # dayctr tracks the index of the observation within the raw 1-d data that we want to select

    dat = da.data

    while yr_idx < numyrs:

        # check if leap year:
        if ((yr - 1900) % 4) == 0:
                n_daysinyear = 366
        else: 
                n_daysinyear = 365


#        print(yr_idx, yr, n_daysinyear)

        if yr_idx == 0:

            day_of_year_idx = firstday_idx

            # in year 0: determine how many days worth of data to select (depends on when dataset starts)
            n_days2select = (n_daysinyear - firstday_idx)

        if yr_idx > 0: 

                day_of_year_idx = 0

                # assume that every year besides first and last has a full 365 or 366 entries
                # (even if some are NaN)
                n_days2select = n_daysinyear

                # if we are in the last year of data:
                if yr_idx == (numyrs-1):

                    # only want to select the days exist in data, in other words, this guards against
                    # the case that the final year of data lacks a full year of entries (ends before 12/31). 
                    n_days2select = (len(dat) - dayctr)


#        print(yr_idx, yr, n_daysinyear, firstday_idx, n_days2select)

        # select data
        year_sel = dat[dayctr:(dayctr+n_days2select)]

        # put it into the array
        yr_day_2darr[yr_idx, (day_of_year_idx):(day_of_year_idx+n_days2select)] = year_sel

        # advance day index, year index 
        dayctr = (dayctr + n_days2select)
        yr+=1
        yr_idx+=1

    
    # put into xr.dataArray
    yr_day_2d_da = xr.DataArray(yr_day_2darr, dims=['year', 'day'], 
                                coords={'year':yrlist, 'day':np.array(range(1, 367))})
    
    return yr_day_2d_da

def detrend_dim_smoothedv2(da, da_roll, dim, deg=1):
    """
    From: https://gist.github.com/rabernat/1ea82bb067c3273a6166d1b1f77d490f
    
    UPDATED 11/20/25 to allow for the passing of 2 arrays: da and da_roll, 
    where da_roll is a smoothed version of da, and we use da_roll to compute the 
    linear trends in each day. We then use these (possibly-less-noisy/unreliable?) trends
    which were computed using the smoother data to detrend the observed "da."
    """
    
    # detrend along a single dimension
    p = da_roll.polyfit(dim=dim, deg=deg) # note using da_roll here
    fit = xr.polyval(da_roll[dim], p.polyfit_coefficients)
    
    return da - fit # subtracting the fit from the NORMAL (unsmoothed) data

def anomalize_dailydata_via_lineartrend_roll11(da):
    """
    this function combines the above two (reshape_1d_to_2d_331) and (detrend_dim)
    plus has to do some pesky reshaping
    
    - with update 11/20/25, generates and passes 11-day rolling data to use for the detrending
    
    3/31/24 - Last updated 11/20/25
    """
    
    # save some original values
    originaldates = da.time
    firstday_num = da.time.dt.dayofyear[0].item()
    firstday_idx = firstday_num - 1
    numdates = len(da)

    ## turn the data from 1-d to 2-d for ease of detrending ##
    da_2d = reshape_1d_to_2d_331(da)
    
    ## also generate 11-day rolling data, reshape that to 2d (11/20/25)
    da_2d_r11 = reshape_1d_to_2d_331(da.rolling(time=11, center=True, min_periods=9).mean()) # min periods decision sorta backed up by data...
    
    ## detrend ##
    da_2d_detrended = detrend_dim_smoothedv2(da=da_2d, da_roll=da_2d_r11, dim='year', deg=1)
    
    # now I have 2-d data that is detrended, however, there are more entries than in 
    # the original input "da" because I've added a bunch of blank spaces for day no. 366 in all the 
    # non-leap years. 

    # this mask identifies dates that have nodata & day num = 366
    empty_leapmask = (((da_2d_detrended.year-1900)%4)>0) * (da_2d_detrended.day==366)
    
    # flatten the 2-d detrended data and the mask
    datflat = da_2d_detrended.data.reshape(-1)
    empty_leapmask_flat = empty_leapmask.data.reshape(-1)

    # mask out the NaN's that came from the leap year column
    datflat_drop_leapNaNs = datflat[~empty_leapmask_flat]

    # cut out any leading or trailing NaN's that could've happened due to incomplete data 
    final_detrened_datflat = datflat_drop_leapNaNs[firstday_idx:(firstday_idx+numdates)]

    # put into dataArray
    detrended_flat_da = xr.DataArray(final_detrened_datflat, dims='time', coords={'time':originaldates})
    
    return detrended_flat_da

def gen_11d_smoothed_stds(da):
    '''
    11/20/25
    '''

    # first, gen standard deviations for every day of year (individually)
    stds = da.groupby('time.dayofyear').std()

    ## apply 11-day rolling window to smooth daily S-D's

    # i need to extend the 'dayofyear' dim in order to get the rolling val's for the 11days near the start b/c otherwise they will be NaN
    ext_arr = np.zeros(shape=400)
    ext_arr[:366] = stds
    ext_arr[366:] = stds[:(400-366)] # beyond day 366, fill in with the beginning of the year again

    # create the 11-day rolling using the extended array
    ext_arr_da = xr.DataArray(ext_arr, dims='dayofyear', coords={'dayofyear':np.array(range(400))+1})
    ext_arr_da11roll = ext_arr_da.rolling(dayofyear=11, center=True).mean()

    # fill this back in on a 366 day-long array
    rolling_arr = np.zeros(shape=366)
    rolling_arr[:] = ext_arr_da11roll[:366]
    rolling_arr[:11] = ext_arr_da11roll[366:377]

    # put back into an xr.DataArray
    stds_11roll_da = xr.DataArray(rolling_arr, dims='dayofyear', coords={'dayofyear':stds.dayofyear})

    return stds_11roll_da

#*#*#
script = 'code/dataprep/prep_daily_stationdata_expanded_slp_roll11.py'
#*#*#

clim_pd = ['1990-01-01', '2023-12-31']
period = clim_pd # (for now)

# dataframe with the stations we are using
df = pd.read_csv('/home/nsiegert/projects/coastal_sst/data/hadisd_stations_using_Expanded.csv')
df = df.drop(['Unnamed: 0'], axis=1)
stalist = df.STAID
df

# open the heatwave ds
hw_ds = xr.open_dataset('/dx02/data/nsiegert/coastal_mhw_data/ALLSTATIONS.1.5deg.marineheatwaves.nc')
hw_ds_time0 = hw_ds.time[0]
hw_ds_time1 = hw_ds.time[-1]
hw_ds_n_times = len(hw_ds.time)


## Prep data for each station ##

print('prepping data for each station. 1.', flush=True)

# output arrays
var_arr = np.zeros(shape=hw_ds.MHW.shape) * np.NaN
var_det_arr = np.zeros(shape=hw_ds.MHW.shape) * np.NaN
var_STdet_arr = np.zeros(shape=hw_ds.MHW.shape) * np.NaN

messed_up_data_staids = []

# for each station,
for i, staid in enumerate(stalist):
#    print(i, staid)

    ## open this station's weather data ##
    # file format: hadisd.3.4.1.2024f_19310101-20250101_010014-99999.nc.gz
    # opening until 1/1/24 b/c of annoying issues with hourly data & date cutoffs. 
    sta_ds = xr.open_dataset('/dx01/nsiegert/HadISD/raw_expanded/hadisd.3.4.1.2024f_19310101-20250101_{}.nc'.format(staid)).sel(time=slice('1990-01-01', '2024-01-01'))
        # why am I not just slicing until '2023-12-31'? must it have something to do with date format or somthing? otherwise it's just pointless...
        
    #*#*#
    # gen daily SLP
    da = sta_ds.slp.where(sta_ds.slp > -2e30).resample(time='D').mean().compute()

    # perform linear anomaly detrending (new version 11/20/25)
    detrend_anom_da = anomalize_dailydata_via_lineartrend_roll11(da[:-1])
    
    ## Compute Std. Anomalies ## (new section 11/20/25)
    detrend_stanom = (detrend_anom_da.groupby('time.dayofyear') / gen_11d_smoothed_stds(da=da))
    
    # save some attributes on first run thru
    if i==0:
        attrs = da.attrs
        attrs['long_name'] = 'daily average Reported Sea Level Pressure at screen height (~2m)'

    #*#*#
    
    # check length of station data that I want:
    missing_flag = 0
    if not len(da[:-1])==hw_ds_n_times:
        print('UH OH, ISSUES WITH STATION DATA LENGTH. LEN = {}, HWDS LEN = {}'.format(len(da[:-1]), hw_ds_n_times))
        missing_flag = 1
    
    ## Put the data into output arrays.     
    if missing_flag: 
        
        messed_up_data_staids.append(staid)
        
        # compute how many days worth of data we need to pad on the front and end (need to subtract 1 for whatever reason from start_pad)
        start_pad = len(pd.date_range(start='1990-01-01', end=pd.to_datetime(da.time[0].data), freq='D')) - 1
        end_pad = len(pd.date_range(start=pd.to_datetime(da.time[-2].data), end='2023-12-31', freq='D')) - 1
        
        print('start pad: {}, end pad: {}'.format(start_pad, end_pad), flush=True)

        #if we have to pad both:
        if start_pad and end_pad:
#            print('both')
            var_arr[i, start_pad:-end_pad] = da[:-1].data
            var_det_arr[i, start_pad:-end_pad] = detrend_anom_da.data
            var_STdet_arr[i, start_pad:-end_pad] = detrend_stanom.data
            
        # if we only have to pad the front:
        elif not end_pad:
 #           print('just start')
            var_arr[i, start_pad:] = da[:-1].data
            var_det_arr[i, start_pad:] = detrend_anom_da.data
            var_STdet_arr[i, start_pad:] = detrend_stanom.data
            
        #if we only have to pad the end:
        elif not start_pad:
#            print('just end')
            var_arr[i, :-end_pad] = da[:-1].data
            var_det_arr[i, :-end_pad] = detrend_anom_da.data
            var_STdet_arr[i, :-end_pad] = detrend_stanom.data
            
    else:    
        # put into arrays.
        var_arr[i, :] = da[:-1].data
        var_det_arr[i, :] = detrend_anom_da.data
        var_STdet_arr[i,:] = detrend_stanom.data
        
    # print number b/c impatient. 
    if i%10==0: print(i, flush=True)
    
print('done with station datasets.', flush=True)    

# put into DataArrays,
var_da = xr.DataArray(var_arr, dims=hw_ds.dims, coords=hw_ds.coords, name=varname)
var_det_da = xr.DataArray(var_det_arr, dims=hw_ds.dims, coords=hw_ds.coords, name=varname)
var_STdet_da = xr.DataArray(var_STdet_arr, dims=hw_ds.dims, coords=hw_ds.coords, name=varname)

# add attrs,
var_da.attrs = attrs
var_det_da.attrs = attrs
var_STdet_da.attrs = attrs

var_det_da.attrs['desc.'] = 'Data anomalized by removing the linear trend from each day of the year, linear trends generated using data smoothed with a centered 11-day rolling window'

var_STdet_da.attrs['desc.'] = 'Data anomalized by removing the linear trend from each day of the year, linear trends generated using data smoothed with a centered 11-day rolling window. Standard deviations generated for each day of the year, then smoothed with an 11-day rolling window as well. Stanom = detrend_anom / rolling_std.'

now = datetime.now()

for da2 in [var_da, var_det_da, var_STdet_da]:
    da2.attrs['script'] = script
    da2.attrs['timestamp'] = now.strftime("%Y-%m-%d %H:%M:%S")

#*#*#
# save
print('saving.', flush=True)
#var_da.to_dataset().to_netcdf('/dx02/data/nsiegert/coastal_mhw_data/ALLSTATIONS.{}.nc'.format(varname))
var_det_da.to_dataset().to_netcdf('/dx02/data/nsiegert/coastal_mhw_data/ALLSTATIONS.{}.detrend_anom.roll11.nc'.format(varname))
var_STdet_da.to_dataset().to_netcdf('/dx02/data/nsiegert/coastal_mhw_data/ALLSTATIONS.{}.detrend_stanom.roll11.nc'.format(varname))

print('dishes are done.', flush=True)

print(messed_up_data_staids, flush=True)