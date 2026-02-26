#!/usr/bin/env python
# coding: utf-8

#---
## This script generates daily Tmin, Tmax & Dewpoint for ***1400+*** selected stations, and generates detrended anomalies
#### Noel Siegert, 1/14/25
#---

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

def detrend_dim(da, dim, deg=1):
    """
    From: https://gist.github.com/rabernat/1ea82bb067c3273a6166d1b1f77d490f
    """
    
    # detrend along a single dimension
    p = da.polyfit(dim=dim, deg=deg)
    fit = xr.polyval(da[dim], p.polyfit_coefficients)
    return da - fit

def anomalize_dailydata_via_lineartrend(da):
    """
    this function combines the above two (reshape_1d_to_2d_331) and (detrend_dim)
    plus has to do some pesky reshaping
    
    3/31/24
    """
    
    # save some original values
    originaldates = da.time
    firstday_num = da.time.dt.dayofyear[0].item()
    firstday_idx = firstday_num - 1
    numdates = len(da)

    ## turn the data from 1-d to 2-d for ease of detrending ##
    da_2d = reshape_1d_to_2d_331(da)

    ## detrend ##
    da_2d_detrended = detrend_dim(da_2d, dim='year', deg=1)

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

script = 'code/dataprep/prep_daily_stationdata_expanded_txtntd.py'

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
tx_arr = np.zeros(shape=hw_ds.MHW.shape) * np.NaN
tn_arr = np.zeros(shape=hw_ds.MHW.shape) * np.NaN
td_arr = np.zeros(shape=hw_ds.MHW.shape) * np.NaN

tx_det_arr = np.zeros(shape=hw_ds.MHW.shape) * np.NaN
tn_det_arr = np.zeros(shape=hw_ds.MHW.shape) * np.NaN
td_det_arr = np.zeros(shape=hw_ds.MHW.shape) * np.NaN

messed_up_data_staids = []

# for each station,
for i, staid in enumerate(stalist):
#    print(i, staid)

    ## open this station's weather data ##
    # file format: hadisd.3.4.1.2024f_19310101-20250101_010014-99999.nc.gz
    # opening until 1/1/24 b/c of annoying issues with hourly data & date cutoffs. 
    sta_ds = xr.open_dataset('/dx01/nsiegert/HadISD/raw_expanded/hadisd.3.4.1.2024f_19310101-20250101_{}.nc'.format(staid)).sel(time=slice('1990-01-01', '2024-01-01'))

    # gen daily Tmin and Tmax
    temps = sta_ds.temperatures.where(sta_ds.temperatures > -2e30)
    tmin_da = temps.resample(time='D').min().compute()
    tmax_da = temps.resample(time='D').max().compute()

    # gen mean daily dewpoint dewpoint
    Td_da = sta_ds.dewpoints.where(sta_ds.dewpoints > -2e30).resample(time='D').mean().compute()

    # perform linear anomaly detrending
    tx_detrend_anom = anomalize_dailydata_via_lineartrend(tmax_da[:-1])
    tn_detrend_anom = anomalize_dailydata_via_lineartrend(tmin_da[:-1])
    td_detrend_anom = anomalize_dailydata_via_lineartrend(Td_da[:-1])

    # save some attributes on first run thru
    if i==0:
        tmaxattrs = tmax_da.attrs
        tmaxattrs['long_name'] = 'Maximum daily temperature generated from hourly Dry bulb air temperature at screen height (~2m)'

        tminattrs = tmin_da.attrs
        tminattrs['long_name'] = 'Minimum daily temperature generated from hourly Dry bulb air temperature at screen height (~2m)'

        tdattrs = Td_da.attrs
        tdattrs['long_name'] = 'Daily Mean Td, generated from hourly Dew point temperature at screen height (~2m)'


    # check length of tmin, tmax, td data that I want:
    missing_flag = 0
    if not len(tmax_da[:-1])==hw_ds_n_times:
        print('UH OH, ISSUES WITH TMIN/MAX DATA LENGTH. LEN = {}, HWDS LEN = {}'.format(len(tmax_da[:-1]), hw_ds_n_times))
        missing_flag = 1
    if not len(Td_da[:-1])==hw_ds_n_times:
#        print('UH OH, ISSUES WITH TMIN/MAX DATA LENGTH. LEN = {}, HWDS LEN = {}'.format(len(Td_da[:-1]), hw_ds_n_times))
        missing_flag = 1
        
    ## Put the data into output arrays.     
    if missing_flag: 
        
        messed_up_data_staids.append(staid)
        
        # compute how many days worth of data we need to pad on the front and end (need to subtract 1 for whatever reason from start_pad)
        start_pad = len(pd.date_range(start='1990-01-01', end=pd.to_datetime(tmax_da.time[0].data), freq='D')) - 1
        end_pad = len(pd.date_range(start=pd.to_datetime(tmax_da.time[-2].data), end='2023-12-31', freq='D')) - 1
        
        print('start pad: {}, end pad: {}'.format(start_pad, end_pad), flush=True)

        #if we have to pad both:
        if start_pad and end_pad:
#            print('both')
            tx_arr[i, start_pad:-end_pad] = tmax_da[:-1].data
            tn_arr[i, start_pad:-end_pad] = tmin_da[:-1].data
            td_arr[i, start_pad:-end_pad] = Td_da[:-1].data
            tx_det_arr[i, start_pad:-end_pad] = tx_detrend_anom.data
            tn_det_arr[i, start_pad:-end_pad] = tn_detrend_anom.data
            td_det_arr[i, start_pad:-end_pad] = td_detrend_anom.data

        # if we only have to pad the front:
        elif not end_pad:
 #           print('just start')
            tx_arr[i, start_pad:] = tmax_da[:-1].data
            tn_arr[i, start_pad:] = tmin_da[:-1].data
            td_arr[i, start_pad:] = Td_da[:-1].data
            tx_det_arr[i, start_pad:] = tx_detrend_anom.data
            tn_det_arr[i, start_pad:] = tn_detrend_anom.data
            td_det_arr[i, start_pad:] = td_detrend_anom.data

        #if we only have to pad the end:
        elif not start_pad:
#            print('just end')
            tx_arr[i, :-end_pad] = tmax_da[:-1].data
            tn_arr[i, :-end_pad] = tmin_da[:-1].data
            td_arr[i, :-end_pad] = Td_da[:-1].data
            tx_det_arr[i, :-end_pad] = tx_detrend_anom.data
            tn_det_arr[i, :-end_pad] = tn_detrend_anom.data
            td_det_arr[i, :-end_pad] = td_detrend_anom.data
            
    else:    
        # put into arrays.
        tx_arr[i, :] = tmax_da[:-1].data
        tn_arr[i, :] = tmin_da[:-1].data
        td_arr[i, :] = Td_da[:-1].data
        tx_det_arr[i, :] = tx_detrend_anom.data
        tn_det_arr[i, :] = tn_detrend_anom.data
        td_det_arr[i, :] = td_detrend_anom.data
        
    # print number b/c impatient. 
    if i%10==0: print(i, flush=True)
    
print('done with station datasets.', flush=True)    


# put into DataArrays,
tx_da = xr.DataArray(tx_arr, dims=hw_ds.dims, coords=hw_ds.coords, name='Tx')
tn_da = xr.DataArray(tn_arr, dims=hw_ds.dims, coords=hw_ds.coords, name='Tn')
td_da = xr.DataArray(td_arr, dims=hw_ds.dims, coords=hw_ds.coords, name='Td')

tx_det_da = xr.DataArray(tx_det_arr, dims=hw_ds.dims, coords=hw_ds.coords, name='Tx')
tn_det_da = xr.DataArray(tn_det_arr, dims=hw_ds.dims, coords=hw_ds.coords, name='Tn')
td_det_da = xr.DataArray(td_det_arr, dims=hw_ds.dims, coords=hw_ds.coords, name='Td')

# add attrs,
tx_da.attrs = tmaxattrs
tn_da.attrs = tminattrs
td_da.attrs = tdattrs

tx_det_da.attrs = tmaxattrs
tn_det_da.attrs = tminattrs
td_det_da.attrs = tdattrs

for da1 in [tx_det_da, tn_det_da, td_det_da]:
    da1.attrs['desc.'] = 'Data anomalized by removing the linear trend from each day of the year'

now = datetime.now()

for da2 in [tx_da, tn_da, td_da, tx_det_da, tn_det_da, td_det_da]:
    da2.attrs['script'] = script
    da2.attrs['timestamp'] = now.strftime("%Y-%m-%d %H:%M:%S")
    
# save
print('saving.', flush=True)
tx_da.to_dataset().to_netcdf('/dx02/data/nsiegert/coastal_mhw_data/ALLSTATIONS.tx.nc')
tn_da.to_dataset().to_netcdf('/dx02/data/nsiegert/coastal_mhw_data/ALLSTATIONS.tn.nc')
td_da.to_dataset().to_netcdf('/dx02/data/nsiegert/coastal_mhw_data/ALLSTATIONS.td.nc')

tx_det_da.to_dataset().to_netcdf('/dx02/data/nsiegert/coastal_mhw_data/ALLSTATIONS.tx.detrend_anom.nc')
tn_det_da.to_dataset().to_netcdf('/dx02/data/nsiegert/coastal_mhw_data/ALLSTATIONS.tn.detrend_anom.nc')
td_det_da.to_dataset().to_netcdf('/dx02/data/nsiegert/coastal_mhw_data/ALLSTATIONS.td.detrend_anom.nc')

print('dishes are done.', flush=True)

print(messed_up_staids, flush=True)