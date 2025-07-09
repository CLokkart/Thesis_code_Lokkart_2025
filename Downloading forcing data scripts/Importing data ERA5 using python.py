#OPTIE 1: Laad de ERA5 data uit de ARCO database en sla ze zelf op in de gewenste bestandsvorm

#Nadeel: het moet nog omgezet worden naar een .nc bestand om op te kunnen slaan

#installeer de volgende modules eerst: fsspec, zarr, xarray

#doe dit als volgt om alle modules in een keer te installeren: "pip install fsspec zarr gcsfs xarray"

#Mocht pip niet herkend worden als command run dan eerst "conda install pip" mogelijk verhelpt dit het probleem

import xarray
from datetime import datetime, timedelta

# Load the the ERA5 dataset from ARCO 
ds = xarray.open_zarr(
    'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',
    chunks=None,
    storage_options=dict(token='anon'))

start_time = datetime(2013, 1, 1, 0)
end_time = datetime(2013, 1, 1, 2)

time_slice = slice(start_time, end_time)

time =      ds.sel(time=time_slice)['time']
levels =    ds.sel(time=time_slice)['level']
P =         ds.sel(time=time_slice)['total_precipitation'].values
E =         ds.sel(time=time_slice)['evaporation'].values
lats =      ds.sel(time=time_slice)['latitude'].values
lons =      ds.sel(time=time_slice)['longitude'].values
u =         ds.sel(time=time_slice)['u_component_of_wind'].values
v =         ds.sel(time=time_slice)['v_component_of_wind'].values
q =         ds.sel(time=time_slice)['specific_humidity'].values
w =         ds.sel(time=time_slice)['vertical_velocity'].values
PW =        ds.sel(time=time_slice)['total_column_water'].values


#%%OPTIE 2: Gebruik de API van CDS store (gratis account vereist)

#voor uitleg hoe je deze kan installeren en werkend kan krijgen zie de volgende link:
#   https://cds.climate.copernicus.eu/how-to-api
#Ik heb zelf nog niet met deze API gewerkt maar volgens mij is hij redelijk eenvoudig

import cdsapi

dataset = "derived-era5-pressure-levels-daily-statistics"
request = {
    "product_type": "reanalysis",
    "variable": [
        "specific_humidity",
        "u_component_of_wind",
        "v_component_of_wind"
    ],
    "year": "2020",
    "month": ["01"],
    "day": ["01"],
    "pressure_level": ["500", "975", "1000"],
    "daily_statistic": "daily_mean",
    "time_zone": "utc+00:00",
    "frequency": "1_hourly"
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()



