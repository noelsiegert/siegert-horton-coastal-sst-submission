#!/usr/bin/env python
# coding: utf-8

# ---
# THIS SCRIPT DOWNLOADS ERA5 DAILY MEAN SINGLE-LEVEL REANALYSIS FROM 1/1/1990 - 12/31/2023
## provide the data that you want in the python arguments from the command line.
## it downloads each month's data one-by-one. (otherwise it takes forever, I thinK??)

import cdsapi
import sys
import pandas as pd
import numpy as np

print('----------------------------')
ds_input_arg = sys.argv[1]
print('System argument: {}'.format(ds_input_arg))
print('----------------------------')
output_filepath = '/dx03/data/ERA5/{}_dailymean'.format(ds_input_arg)


# for every year, month in our dataset
for year in range(1990,2024):
    y_str = str(year)

    for mo in range(1, 13): 
        mo_str = str(mo).zfill(2)
        
        # get no. days in this month
        p = pd.Period('{}-{}-1'.format(year, mo))
        n_days = p.days_in_month
        days_list = [str(i) for i in range(1, n_days+1)]

        ### Perform the ERA5 CDSAPI REQUEST ###
        dataset = "derived-era5-single-levels-daily-statistics"
        request = {
            "product_type": "reanalysis",
            "variable": [ds_input_arg],
            "year": y_str,
            "month": [mo_str],
            "day": days_list,
            "daily_statistic": "daily_mean",
            "time_zone": "utc+00:00",
            "frequency": "1_hourly",
            "area": [70, -180, -58, 180]
        }
        target = '{}/{}_dailymean_{}{}.nc'.format(output_filepath, ds_input_arg, y_str, mo_str)

        client = cdsapi.Client()
        client.retrieve(dataset, request, target)

        print('{} downloaded!'.format(target))

    
print('dishes are done.')
