#!/usr/bin/env python
# coding: utf-8

# ---
# ## This script will input a a prepared PROFILE dataset (12 levels), and generate detrended anom's and std. anom's.
# ##### (using the 11-day rolling method for detrending) -- USER supplies the varname
# ##### related to detrend_stanom_vars_roll11.py on Deela
# #### Noel Siegert, 12/1/25
# ---

# kernel: enso_imp

# imports
import os
import xarray as xr
import numpy as np
import netCDF4 
import glob
import pandas as pd
from datetime import datetime
from scipy import stats
import sys


# read the variable name from input arg's
print('----------------------------')
ds_input_arg = sys.argv[1]
varname = ds_input_arg # should be either 'T' or 'Q'
var = varname
print('System argument (Varname): {}'.format(ds_input_arg))
print('----------------------------')


# function defs
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


script = os.getcwd() + '/detrend_stanom_VARprofile_roll11.py'

# dataframe with the stations we are using
df = pd.read_csv('/glade/u/home/nsiegert/projects/coastal_sst/data/hadisd_stations_using_Expanded.csv')
df = df.drop(['Unnamed: 0'], axis=1)

# domains
domain_list = ['1.5deg', 'OverStations']


for domain in domain_list:

    # open this domain's raw <var> profile data
    var_profile_da = xr.open_dataset('/glade/u/home/nsiegert/projects/coastal_sst/data/ALLSTATIONS.{}.{}profile.nc'.format(domain, var))[var]

    # lists for concatenating dataArrays later. 
    Profile_anomda_list = []
    Profile_STanomda_list = []


    # for each of the 12 levels:
    for lvl in var_profile_da.level:
    
        print('DOMAIN: {} LEVEL: {}'.format(domain, str(int(lvl))), flush=True)
    
        # select just one level in order to anomalize and save intermediate file as well. 
        var_da = var_profile_da.sel(level=lvl)
    
        # output arrays
        var_det_arr = np.zeros(shape=var_da.shape) * np.nan
        var_STdet_arr = np.zeros(shape=var_da.shape) * np.nan
    
        # for each station...
        for stanum in range(len(df)):
    
            if stanum%100==0:
                print(stanum)
    
            # select this station's raw data
            sta_da = var_da[stanum, :]
    
            # perform linear anomaly detrending (new version 11/20/25)
            detrend_anom_da = anomalize_dailydata_via_lineartrend_roll11(sta_da)
    
            ## Compute Std. Anomalies ## (new section 11/20/25)
            detrend_stanom = (detrend_anom_da.groupby('time.dayofyear') / gen_11d_smoothed_stds(da=sta_da))
    
            # save some attributes on first run thru
            if stanum==0:
                attrs = var_da.attrs
    
            # put into the output arrays at that station's position
            var_det_arr[stanum, :] = detrend_anom_da.data
            var_STdet_arr[stanum, :] = detrend_stanom.data
    
        # convert into DataArrays,
        var_det_da = xr.DataArray(var_det_arr, dims=var_da.dims, coords=var_da.coords, name=varname)
        var_STdet_da = xr.DataArray(var_STdet_arr, dims=var_da.dims, coords=var_da.coords, name=varname)
    
        # add attrs
        var_det_da.attrs = attrs
        var_STdet_da.attrs = attrs
        var_det_da.attrs['desc.'] = 'Data anomalized by removing the linear trend from each day of the year, linear trends generated using data smoothed with a centered 11-day rolling window'
        var_STdet_da.attrs['desc.'] = 'Data anomalized by removing the linear trend from each day of the year, linear trends generated using data smoothed with a centered 11-day rolling window. Standard deviations generated for each day of the year, then smoothed with an 11-day rolling window as well. Stanom = detrend_anom / rolling_std.'
    
        now = datetime.now()
    
        for da2 in [var_det_da, var_STdet_da]:
            da2.attrs['script'] = script
            da2.attrs['timestamp'] = now.strftime("%Y-%m-%d %H:%M:%S")
    
        # save
        print('saving. THIS LEVEL', flush=True)
    
        var_det_da.to_dataset().to_netcdf('/glade/u/home/nsiegert/projects/coastal_sst/data/ALLSTATIONS.{}.{}profile.lv{}.detrend_anom.roll11.CAN_DELETE.nc'.format(domain, varname, str(int(lvl))))
        var_STdet_da.to_dataset().to_netcdf('/glade/u/home/nsiegert/projects/coastal_sst/data/ALLSTATIONS.{}.{}profile.lv{}.detrend_stanom.roll11.CAN_DELETE.nc'.format(domain, varname, str(int(lvl))))
    
        # also save in list 
        Profile_anomda_list.append(var_det_da)
        Profile_STanomda_list.append(var_STdet_da)

    # after having completed all levels, concatenate and save as well (these are the files I will keep)
    var_profile_det_da = xr.concat(Profile_anomda_list, dim='level')
    var_profile_STdet_da = xr.concat(Profile_STanomda_list, dim='level')
    
    print('saving. THIS Domain ({}), ALL Levels'.format(domain), flush=True)
    
    var_profile_det_da.to_dataset().to_netcdf('/glade/u/home/nsiegert/projects/coastal_sst/data/ALLSTATIONS.{}.{}profile.detrend_anom.roll11.nc'.format(domain, varname))
    var_profile_STdet_da.to_dataset().to_netcdf('/glade/u/home/nsiegert/projects/coastal_sst/data/ALLSTATIONS.{}.{}profile.detrend_stanom.roll11.nc'.format(domain, varname))

print('Dishes are done.', flush=True)