#!/bin/bash
# This script downloads daily OISST data from: 1/2024 thru 5/2024 (June / July files are currently "preliminary")
# Noel Siegert, 7/2/2024

# data info: https://www.ncei.noaa.gov/products/optimum-interpolation-sst
# data source: https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/

# download directory: /dx01/data/OISST/raw

# template wget command:
# wget -r -nH --cut-dirs=6 --no-parent --reject="index.html*" --reject="*.txt"  https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/202008/


for m in 01 02 03 04 05
do

    # print statement
    echo $m
    # download the data
    wget -r -nH --cut-dirs=6 --no-parent --reject="index.html*" --reject="*.txt"  https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/2024$m/

done

# for some reason there is a file 'robots.txt' that gets downloaded
rm -f robots.txt

# now we are done
echo 'dishes are done.'