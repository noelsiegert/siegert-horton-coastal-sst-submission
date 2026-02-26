# ---
# ## Function Defs for analysis for the Coastal project.
# #### Noel Siegert, 2/14/25
# #### Last Updated: 10/16/25
# ---

# dependencies
import os
import xarray as xr
import numpy as np
import netCDF4 
import glob
import pandas as pd
import geopandas as gpd
from datetime import datetime
from scipy import stats


# bin for lat scatters
def bin_lat_scatters(latbins, lats, dat):
    '''
    Params 
        - latbins (np.array) latitude bins that you want to use
        - lats (np.array) list of lat coords for your data
        - dat (np.array) the data you want to bin by lat
        
    Returns
        - binned_means (list) average <dat> value within each lat bin
        - bin_midpoints (list) midway lat point of each bin (for plotting)
        - bin_n_obs (list) n. observations that fall within each bin.
        
    update: 5/9/25 to skip any inf values (comes up when I standardize...)
    '''
    
    binned_means = []
    bin_midpoints = []
    bin_n_obs = []

    i = 0
    
    no_inf_mask = np.logical_not(np.isinf(dat)) # 5/9

    while i < len(latbins)-1:

        bot_bin = latbins[i]
        top_bin = latbins[i+1]

    #    print(bot_bin, top_bin)

        binmask = (lats >= bot_bin) * (lats < top_bin)

        binned_means.append(np.nanmean(dat[binmask*no_inf_mask]))
        bin_midpoints.append(((bot_bin+top_bin)/2))
        bin_n_obs.append(np.sum(binmask))

        i += 1
        
    return binned_means, bin_midpoints, bin_n_obs


# make zonally-grouped anomalies (express dat. as anomaly from its binned zonal mean
def gen_zonal_anoms(latbins, lats, dat):
    """
    2/12/25
    
    Give it some data (dat), lat bins (latbins), and lat values that correspond to data points (lats),
    and it will give you each data point expressed as the anomaly from the zonal bin-avg for the bin it falls in.
    
    """
    zonal_anom_dat = np.zeros(shape=dat.shape) * np.nan

    binned_means = []
    bin_midpoints = []
    bin_n_obs = []

    i = 0

    while i < len(latbins)-1:

        bot_bin = latbins[i]
        top_bin = latbins[i+1]

        # gen mask for the lat bin
        binmask = (lats >= bot_bin) * (lats < top_bin)

        # gen mean <var> within this bin, across all longitudes
        mean_val_this_bin = np.nanmean(dat[binmask])

        # gen anomaly 
        zonal_anom_dat[binmask] = dat[binmask] - mean_val_this_bin

        binned_means.append(mean_val_this_bin)
        bin_midpoints.append(((bot_bin+top_bin)/2))
        bin_n_obs.append(np.sum(binmask))

        i += 1

    return zonal_anom_dat, binned_means, bin_midpoints, bin_n_obs

# As above but with percentile
def gen_zonal_pctiles(latbins, lats, dat):
    """
    2/12/25
    
    Give it some data (dat), lat bins (latbins), and lat values that correspond to data points (lats),
    and it will give you each data point expressed as the percentile compared to the obs in the zonal bin it falls in.
    
    """
    zonal_pctile_dat = np.zeros(shape=dat.shape) * np.nan

    bin_midpoints = []
    bin_n_obs = []

    i = 0

    while i < len(latbins)-1:

        bot_bin = latbins[i]
        top_bin = latbins[i+1]

        # gen mask for the lat bin
        binmask = (lats >= bot_bin) * (lats < top_bin)

        # gen percentiles 
        zonal_pctile_dat[binmask] = stats.percentileofscore(dat[binmask], dat[binmask])

        bin_midpoints.append(((bot_bin+top_bin)/2))
        bin_n_obs.append(np.sum(binmask))

        i += 1

    return zonal_pctile_dat, bin_midpoints, bin_n_obs



def gen_zonal_stanoms(latbins, lats, dat):
    """
    3/3/25
    
    Give it some data (dat), lat bins (latbins), and lat values that correspond to data points (lats),
    and it will give you each data point expressed as the standardized anomaly from the zonal bin-avg for the bin it falls in.
    
    """
    zonal_anom_dat = np.zeros(shape=dat.shape) * np.nan

    binned_means = []
    bin_midpoints = []
    bin_n_obs = []

    i = 0

    while i < len(latbins)-1:

        bot_bin = latbins[i]
        top_bin = latbins[i+1]

        # gen mask for the lat bin
        binmask = (lats >= bot_bin) * (lats < top_bin)

        # gen mean <var> within this bin, across all longitudes
        mean_val_this_bin = np.nanmean(dat[binmask])
        
        # gen std <var> within this bin, across all latitudes
        std_this_bin = np.nanstd(dat[binmask])


        # gen anomaly 
        zonal_anom_dat[binmask] = (dat[binmask] - mean_val_this_bin) / std_this_bin

        binned_means.append(mean_val_this_bin)
        bin_midpoints.append(((bot_bin+top_bin)/2))
        bin_n_obs.append(np.sum(binmask))

        i += 1

    return zonal_anom_dat, binned_means, bin_midpoints, bin_n_obs


# originally from 1_30_leadlags.ipynb:
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
        
    return lag_arr

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


def compute_heat_index(T, H):
    """
    This function uses the NWS's algorithm to compute heat index. 
    Source: Fig. 3 of this paper https://ehp.niehs.nih.gov/doi/10.1289/ehp.1206273
    This can be checked via this calculator: https://www.wpc.ncep.noaa.gov/html/heatindex.shtml
    (all the 'conditions' are verified to have worked)
    
    For whatever reason, they compute HI using fahrenheit. 
    
    Params: T (np.array) daily (mean) temperatures
            H (np.array) daily (mean) relative humidity
            
    Returns: HI (np.array) heat index, in degrees F (can convert to ˚C as well if needed)
    """
    
    # array to hold heat index values. 
    HI = np.zeros(shape=T.shape) * np.nan
    remaining_arr = np.ones(shape=T.shape) # this array will track the days that remain without a HI value. 
    
    # array to hold heat index values. 
    HI = np.zeros(shape=T.shape) * np.nan
    remaining_arr = np.ones(shape=T.shape) # this array will track the days that remain without a HI value. 
#    conditions_arr = np.zeros(shape=T.shape) # this array will track the conditions that applied to that HI calculation.

#    print('condition 0: {}'.format(np.sum(remaining_arr)))

    # CONDITION 1: if T <= 40F, HI = T
    HI[T<=40] = T[T <= 40]
    remaining_arr[T<=40] = 0
#    conditions_arr[T<=40] = 1
#    print('condition 1: {}'.format(np.sum(remaining_arr)))

    # otherwise, compute A:
    A = -10.3 + (1.1 * T) + (0.047 * H)

    # CONDITION 2: if A < 79F, HI = A
    HI[A<79] = A[A<79]
    remaining_arr[A<79] = 0
#    conditions_arr[A<79] = 2
#    print('condition 2: {}'.format(np.sum(remaining_arr)))

    # otherwise, compute B: (CAN I DO SOME ROUNDOFF HERE TO SAVE COMPUTING TIME...)
    B = -42.379 + (2.04901523*T) + (10.14333127*H) - (0.22475541*T*H) - (6.83783e-3*(T**2)) - (5.481717e-2*(H**2)) \
        + (1.22874e-3*(T**2)*H) + (8.5282e-4*T*(H**2)) - (1.99e-6*(T**2)*(H**2))

    # CONDITION 3: are H <= 13% and 80 <= T <= 112?
    cond3 = (H<=13)*(80<=T)*(T<=112)

    # if yes to cond. 3,
    cond3B = B - ((13-H)/4) * (((17-(np.abs(T - 95)))/17)**(1/2))
    HI[cond3] = cond3B[cond3]
    remaining_arr[cond3] = 0
#    conditions_arr[cond3] = 3
#    print('condition 3: {}'.format(np.sum(remaining_arr)))

    # CONDITION 4: Are H > 85% and 80 <= T <= 87?
    cond4 = (H>85)*(80<=T)*(T<=87)

    # if yes to cond. 4,
    cond4B = B + 0.02 * (H-85) * (87-T)
    HI[cond4] = cond4B[cond4]
    remaining_arr[cond4] = 0
#    conditions_arr[cond4] = 4
#    print('condition 4: {}'.format(np.sum(remaining_arr)))

    # otherwise, HI = B
    HI[remaining_arr.astype(bool)] = B[remaining_arr.astype(bool)]
#    conditions_arr[remaining_arr.astype(bool)] = 5
    
    # return the heat index
    return HI

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