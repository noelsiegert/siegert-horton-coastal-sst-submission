#!/bin/bash

# This script downloads HadISD weather station data for 1400 stations around the globe (identified arleady)
# 1/14/2025
## HadISD version 3.4.1.202405p
## Info: https://www.metoffice.gov.uk/hadobs/hadisd/v341_202405p/index.html

filename='HadISD_station_urls_Expanded.txt'

# download files
echo 'downloading'

while read p; do 
    wget $p
done < "$filename"

# unzip files
echo 'unzipping'
gunzip *.nc.gz

echo 'dishes are done.'
