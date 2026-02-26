# EDIT 1/14/25
import pandas as pd


# load in current (expanded!) stations 
df = pd.read_csv('/home/nsiegert/projects/coastal_sst/data/hadisd_stations_using_Expanded.csv')

url = 'https://www.metoffice.gov.uk/hadobs/hadisd/v341_2024f/data/hadisd.3.4.1.2024f_19310101-20250101_STAID.nc.gz'

with open("HadISD_station_urls_Expanded.txt", "w") as text_file:

    # for every station:
    for station in df.iterrows():

        staid = station[1]['STAID']

        text_file.write(url.replace('STAID', staid))
        text_file.write('\n')

text_file.close()
