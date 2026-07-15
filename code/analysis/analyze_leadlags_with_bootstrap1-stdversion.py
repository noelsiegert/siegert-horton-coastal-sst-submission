#---
## This script will create evolutions of various variables for the Supplemental Fig's (4 - 5) drafts. I'm using 3˚ latitude bins, updated to add the density-weighted region-stratified bootstrapped CI's
## THIS TIME, using the standardized version of each variable. 
##### 6/18/26 
##### 6/21/26 [re-running this for SLP, add code to NaN out any inf's]
#---

# kernel: pangeo23

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
from scipy.spatial import cKDTree

from coastal_analysis_fns import *

import warnings
warnings.filterwarnings('ignore')


def convert_dat_to_pctile(da):
    '''
    8/18/25
    
    Params:
    - da (DataArray) dims = staid, time
    
    returns:
    - pctile_da_out (xr.dataArray) same dims, data = transformed to percentile, within each station & day of year. 
    '''
    
    # np.arrays for raw data and output percentile-d data.
    dat = da.data
    pctile_dat = np.zeros(shape=dat.shape) * np.nan
    
    # for each day of the year, transform each station's data into percentile relative to that day-of-yr
    for d in range(1, 367):

        # mask to select day of year
        dayofyear_mask = (da.time.dt.dayofyear==d)
        ndays = np.sum(dayofyear_mask).item()

        # compute percentile value for each entry
        h22 = stats.rankdata(dat[:, dayofyear_mask], axis=1)
        h22_pct = (h22/ndays) * 100 # converting rank to percentile

        # save that day-of-year's percentile data
        pctile_dat[:, dayofyear_mask] = h22_pct
        
    # return
    pctile_da_out = xr.DataArray(pctile_dat, dims=da.dims, coords=da.coords)
    
    return pctile_da_out

# for each value in the 'valid' (dayofyearmask), I need to compare that value to all the pooled values in the dayofyearmask11,
# But I want to EXCLUDE the date itself from being assigned. 
def make_lagged_onsetmask1D(dat, lag):
    '''
    This function takes the mhw_onsets array/mask and returns one shifted over "lag" number of days
    
    params:
        dat (np.array or xr.dataArray) the mask you want to shift
        lag (int) the number of days/spaces you want to shift. negative = backwards/before, positive=forwards/after
        
    returns: lag_arr
    
    10/14/25 - adapted for 1-dimensional input data
    '''
    
    
    # empty array
    lag_arr = np.zeros(shape=dat.shape).astype(bool)
    abslag = abs(lag)

    # if a negative lag (aka before event onset)
    if lag < 0:

        lag_arr[:lag] = dat[abslag:]

    elif lag > 0:

        lag_arr[lag:] = dat[:(-lag)]
        
    elif lag==0:
        
        lag_arr = dat # self-explanatory?
        
    return lag_arr

def make_lagged_onsetmask(dat, lag):
    '''
    This function takes the mhw_onsets array/mask and returns one shifted over "lag" number of days
    
    params:
        dat (np.array or xr.dataArray) the mask you want to shift
        lag (int) the number of days/spaces you want to shift. negative = backwards/before, positive=forwards/after
        
    returns: lag_arr
    
    1/30/25
    '''
    
    
    # empty array
    lag_arr = np.zeros(shape=dat.shape).astype(bool)
    abslag = abs(lag)

    # if a negative lag (aka before event onset)
    if lag < 0:

        lag_arr[:, :lag] = dat[:, abslag:]

    elif lag > 0:

        lag_arr[:, lag:] = dat[:, :(-lag)]
        
    elif lag==0:
        lag_arr = dat
        
    return lag_arr

def bin_lat_scatters(latbins, lats, dat):
    
    binned_means = []
    bin_midpoints = []
    bin_n_obs = []

    i = 0

    while i < len(latbins)-1:

        bot_bin = latbins[i]
        top_bin = latbins[i+1]

    #    print(bot_bin, top_bin)

        binmask = (lats >= bot_bin) * (lats < top_bin)

        binned_means.append(np.nanmean(dat[binmask]))
        bin_midpoints.append(((bot_bin+top_bin)/2))
        bin_n_obs.append(np.sum(binmask))

        i += 1
        
    return binned_means, bin_midpoints, bin_n_obs

def prep_global_binned_leadlags(da, event_onsets, window, latbins):

    '''
    Function-izing code from earlier work...
    
    This function creates binned/lagged dataArrays of a desired variable centered on desired events, binned by lat.
    
    Params:
    - da (xr.dataArray) the input variable. dims = [staid, time]
    - event_onsets (xr.dataArray) bool indices to the events of interest. dims = [staid, time]
    - window (int) number of days before and after event onset which you are interested in binning
    - latbins (np.array) latitude bins to use
    
    Returns:
    - var_bin_lag_da (xr.Da) variable, averaged across various lags and binned by lat. dims = [lat bin, event-day]
    
    '''

    # these will hold the MHW evolutions in a different form. shape = [lat bin, lag day]
    bin_lag_profile_arr = np.zeros(shape=((len(latbins)-1), ((window*2)+1)))

    # counter var
    i = 0

    # will take lags from -w days prior to mhw event thru +w days 
    for l in range(-window, (window+1)):

        # make a mask to select all days with lag=l rel. to mhw onsets
        if not l==0:
            lagmhwmask = make_lagged_onsetmask(dat=event_onsets, lag=l)
        elif l==0: 
            lagmhwmask = event_onsets

        # sel avg. <var> at every station at that <event> lag
        var_lag_means = da.where(lagmhwmask).mean(dim='time').to_numpy()

        # bin by lat
        var_lag_means_binned, bin_midpoints, bin_nobs = bin_lat_scatters(latbins=latbins, lats=gdf['LAT'].to_numpy(), dat=var_lag_means)

        # add to arrays
        bin_lag_profile_arr[:, i] = var_lag_means_binned

        i+=1

    # put into dataArray
    bin_lag_profile_da = xr.DataArray(bin_lag_profile_arr, dims=['lat bin', 'day'], coords={'lat bin':bin_midpoints, 'day':np.array(range(-window, (window+1)))})

    return bin_lag_profile_da                      

def prep_global_binned_leadlags_withResampleBoot(da, event_onsets, window, latbins, lats, lons, nboot, radius_deg):

    '''
  ###  Function-izing code from earlier work...
    
 ###   This function creates binned/lagged dataArrays of a desired variable centered on desired events, binned by lat.
    
    Params:
    - da (xr.dataArray) the input variable. dims = [staid, time]
    - event_onsets (xr.dataArray) bool indices to the events of interest. dims = [staid, time]
    - window (int) number of days before and after event onset which you are interested in binning
    - latbins (np.array) latitude bins to use
    
    Returns:
    - var_bin_lag_da (xr.Da) variable, averaged across various lags and binned by lat. dims = [lat bin, event-day]
    
    '''

    # these will hold the MHW evolutions in a different form. shape = [lat bin, lag day, nboot]
    bin_lag_profile_arr = np.zeros(shape=((len(latbins)-1), ((window*2)+1), nboot))

    # counter var
    i = 0

    # will take lags from -w days prior to mhw event thru +w days 
    for l in range(-window, (window+1)):

#        print('window={}'.format(l))

        # make a mask to select all days with lag=l rel. to mhw onsets
        if not l==0:
            lagmhwmask = make_lagged_onsetmask(dat=event_onsets, lag=l)
        elif l==0: 
            lagmhwmask = event_onsets

        # sel avg. <var> at every station at that <event> lag
        var_lag_means = da.where(lagmhwmask).mean(dim='time').to_numpy()

        ### Now that I have each station's mean <var> at lag = l,
        # I can pass that data to the fn which generates bootstrapped means & CI's using density-weighted sampling
        medCIs, hiCIs, loCIs, binned_boots, bin_midpoints, bin_n_obs = bin_lat_scatters_densitywtd_bootCI(dat=var_lag_means, latbins=latbins, lats=lats, lons=lons, nboot=nboot, radius_deg=radius_deg)
            # all I really want is bin_midpoints

        # save into the array
        bin_lag_profile_arr[:, i, :] = binned_boots
        
        # bin by lat
#        var_lag_means_binned, bin_midpoints, bin_nobs = bin_lat_scatters(latbins=latbins, lats=gdf['LAT'].to_numpy(), dat=var_lag_means)

        # add to arrays
 #       bin_lag_profile_arr[:, i] = var_lag_means_binned

        i+=1

    # put into dataArray
    bin_lag_profile_da = xr.DataArray(bin_lag_profile_arr, dims=['lat bin', 'day', 'boot'], 
                                      coords={'lat bin':bin_midpoints, 'day':np.array(range(-window, (window+1))), 'boot':np.array(range(nboot))})

    return bin_lag_profile_da                      


script = 'analyze_leadlags_with_bootstrap1-stdversion.py'

# dataframe with the stations we are using
df = pd.read_csv('/home/nsiegert/projects/coastal_sst/data/hadisd_stations_using_Expanded.csv')
df = df.drop(['Unnamed: 0'], axis=1)

# convert df into geodataframe for ease of plotting
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=df.LON, y=df.LAT))


# open geophysical data

# MHW
hw_ds = xr.open_dataset('/dx02/data/nsiegert/coastal_mhw_data/ALLSTATIONS.1.5deg.marineheatwaves_roll11.nc') # 11 DAY ROLLING!!!
mhw_mask = hw_ds.MHW
mhw_onset_mask = hw_ds.MHW_onsets.astype(bool)

# thw
thw_ds = xr.open_dataset('/dx02/data/nsiegert/coastal_mhw_data/ALLSTATIONS.1.5deg.heatwaves.nc')
thw_mask = thw_ds.THW
thw_onset_mask = thw_ds.THW_onsets.astype(bool)

# open more geophysical data

# station var's
tx_det_da_st = xr.open_dataset('/dx02/data/nsiegert/coastal_mhw_data/ALLSTATIONS.tx.detrend_stanom.roll11.nc').Tx # new file 11/21
tn_det_da_st = xr.open_dataset('/dx02/data/nsiegert/coastal_mhw_data/ALLSTATIONS.tn.detrend_stanom.roll11.nc').Tn # new file 11/21
td_det_da_st = xr.open_dataset('/dx02/data/nsiegert/coastal_mhw_data/ALLSTATIONS.td.detrend_stanom.roll11.nc').Td # new file 11/21
slp_det_da_st = xr.open_dataset('/dx02/data/nsiegert/coastal_mhw_data/ALLSTATIONS.slp.detrend_stanom.roll11.nc').slp # new file 11/21
windspeed_det_da_st = xr.open_dataset('/dx02/data/nsiegert/coastal_mhw_data/ALLSTATIONS.windspeeds.detrend_stanom.roll11.nc').windspeeds # new file 11/21

#  SST 
sst_det_da_st = xr.open_dataset('/dx02/data/nsiegert/coastal_mhw_data/ALLSTATIONS.sst.detrend_stanom.roll11.nc').sst.sel(time=slice('1990-01-01', '2023-12-31')) # new file 11/21

# open heatflux data (anomalies, over the stations)
slhf15_st = xr.open_dataset('/dx02/data/nsiegert/coastal_mhw_data/ALLSTATIONS.1.5deg.slhf.detrend_stanom.roll11.nc').slhf
sshf15_st = xr.open_dataset('/dx02/data/nsiegert/coastal_mhw_data/ALLSTATIONS.1.5deg.sshf.detrend_stanom.roll11.nc').sshf
ssr15_st = xr.open_dataset('/dx02/data/nsiegert/coastal_mhw_data/ALLSTATIONS.1.5deg.ssr.detrend_stanom.roll11.nc').ssr
str15_st = xr.open_dataset('/dx02/data/nsiegert/coastal_mhw_data/ALLSTATIONS.1.5deg.str.detrend_stanom.roll11.nc').str

# open heat fluxes over land as well
slhfSTA_st = xr.open_dataset('/dx02/data/nsiegert/coastal_mhw_data/ALLSTATIONS.OverStations.slhf.detrend_stanom.roll11.nc').slhf
sshfSTA_st = xr.open_dataset('/dx02/data/nsiegert/coastal_mhw_data/ALLSTATIONS.OverStations.sshf.detrend_stanom.roll11.nc').sshf
ssrSTA_st = xr.open_dataset('/dx02/data/nsiegert/coastal_mhw_data/ALLSTATIONS.OverStations.ssr.detrend_stanom.roll11.nc').ssr
strSTA_st = xr.open_dataset('/dx02/data/nsiegert/coastal_mhw_data/ALLSTATIONS.OverStations.str.detrend_stanom.roll11.nc').str

# mixed layer depth
mld_da_st = xr.open_dataset('/dx02/data/nsiegert/coastal_mhw_data/ALLSTATIONS.mld.detrend_stanom.roll11.nc').mld

# season masks
szn_ds = xr.open_dataset('/dx02/data/nsiegert/coastal_mhw_data/ALLSTATIONS.warm_cold_seasons.nc')
warmszn_mask = szn_ds.warmszn_mask.astype(bool)
coldszn_mask = szn_ds.coldszn_mask.astype(bool)


# gather concurrent event onsets
concurrent_ev_mask = (mhw_mask * thw_mask)

# onsets are days where there is no concurrent event (=0), and the next day = 1
concurrent_ev_onsetmaskPre = (concurrent_ev_mask==0).data[:, :-1] * concurrent_ev_mask.data[:, 1:] # and have to drop the last day of ds from analysis. This gets us an onset mask for the second day of the ds onwards. 
concurrent_ev_onsetmask = np.zeros(shape=concurrent_ev_mask.shape)
concurrent_ev_onsetmask[:, 1:] = concurrent_ev_onsetmaskPre # fill in the 0 on the first day0 (which can't be an onset b/c we don't know the status of the day prior)

concurrent_ev_onsetmask_da = xr.DataArray(concurrent_ev_onsetmask, dims=concurrent_ev_mask.dims, coords=concurrent_ev_mask.coords)

## 
# Create and save arrays with shape = [lat bin, lag day, nboot], that I can use for significance testing and global (area-weighted) averaging
##


# set window, and lat bins
ww = 30
lbins = np.arange(-60, 71, 3)

# list of datasets to build a lat-stratified bootstrap, lag, mean array... etc. 
var_to_boot_da_list = [sst_det_da_st, tx_det_da_st, td_det_da_st, slp_det_da_st, windspeed_det_da_st, slhf15_st, sshf15_st, ssr15_st, str15_st, slhfSTA_st, sshfSTA_st, ssrSTA_st, strSTA_st, mld_da_st]

var_to_boot_da_list = [slp_det_da_st] # 6/21


for szn in ['w', 'c']:
        
    for ev_type in ['MHW', 'THW', 'CON']:

        print('{},{}'.format(ev_type,szn))

        # select the appropriate event onset mask based on event type and season
        if ev_type=='MHW':
            if szn=='w':
                eventUse = xr.where((hw_ds.MHW_onsets*np.logical_not(thw_ds.THW)*warmszn_mask), True, False) 
                eventUseTrim = eventUse.sel(time=slice('1993-01-01', '2021-06-30')) # for the MLD data
            elif szn=='c':
                eventUse = xr.where((hw_ds.MHW_onsets*np.logical_not(thw_ds.THW)*coldszn_mask), True, False) 
                eventUseTrim = eventUse.sel(time=slice('1993-01-01', '2021-06-30'))
        elif ev_type=='THW':
            if szn=='w':
                eventUse = xr.where((thw_ds.THW_onsets*np.logical_not(hw_ds.MHW)*warmszn_mask), True, False) 
                eventUseTrim = eventUse.sel(time=slice('1993-01-01', '2021-06-30'))
            elif szn=='c':
                eventUse = xr.where((thw_ds.THW_onsets*np.logical_not(hw_ds.MHW)*coldszn_mask), True, False) 
                eventUseTrim = eventUse.sel(time=slice('1993-01-01', '2021-06-30'))
        elif ev_type=='CON':
            if szn=='w':
                eventUse = xr.where((concurrent_ev_onsetmask_da*warmszn_mask), True, False) 
                eventUseTrim = eventUse.sel(time=slice('1993-01-01', '2021-06-30'))
            elif szn=='c':
                eventUse = xr.where((concurrent_ev_onsetmask_da*coldszn_mask), True, False) 
                eventUseTrim = eventUse.sel(time=slice('1993-01-01', '2021-06-30'))

        
        for varctr, varname in enumerate(['slp']):
        #for varctr, varname in enumerate(['sst', 'tx', 'td', 'slp', 'windspeed', 'slhf', 'sshf', 'ssr', 'str', 'slhfSTA', 'sshfSTA', 'ssrSTA', 'strSTA', 'mld']):

            # select data
            da_to_bin = var_to_boot_da_list[varctr]

            print(varname)

            # for normal, non-MLD cases:
            if not varname=='mld':
    
                # do the bootstrap
                var_binlag_event_da = prep_global_binned_leadlags_withResampleBoot(da=da_to_bin, event_onsets=eventUse.data, window=ww, latbins=lbins, 
                                                 lats=df['LAT'].values, lons=df['LON'].values, nboot=1000, radius_deg=3.0)
                print('normal version')

            # have to do slightly different stuff for MLD
            elif varname =='mld':

                # MLD... (need to trim the event selection)
                da_to_bin['time'] = eventUseTrim.time

                var_binlag_event_da = prep_global_binned_leadlags_withResampleBoot(da=da_to_bin, event_onsets=eventUseTrim.data, window=ww, latbins=lbins, 
                                                 lats=df['LAT'].values, lons=df['LON'].values, nboot=1000, radius_deg=3.0)

                print('mld version.')
            
            ds_out = var_binlag_event_da.to_dataset(name=varname)
            ds_out.attrs['script'] = os.getcwd() + script
            
            now = datetime.now()
            ds_out.attrs['timestamp'] = now.strftime("%Y-%m-%d %H:%M:%S")
            
            savename = '/dx02/data/nsiegert/coastal_mhw_data/{}_std_{}_{}_latbinned_bootmeans.nc'.format(varname, ev_type, szn)
            print('saving: {}'.format(savename))

            # save
            ds_out.to_netcdf(savename)