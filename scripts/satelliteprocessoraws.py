# Uses AWS to download data from GOES-16 satellites (Significantly faster than downloading directly)
#Measurements should be every 10 minutes or so

#Imports 
import os
import re
import numpy as np
import xarray as xr
import s3fs
import warnings
from tqdm import tqdm
from datetime import datetime, timedelta
from pyproj import CRS, Transformer
from collections import defaultdict
from multiprocessing import get_context

warnings.filterwarnings("ignore", message=".*Times can't be serialized faithfully.*", category=UserWarning)

#Changeable Parameters
path_name = 'noaa-goes16/ABI-L1b-RadF' #Get data from full disk satellite (less likely to move)
start_date = datetime(2024, 9, 1) # Where you want your time series to start from 
num_days = 100 #duration of time series
bands = [f'C{b:02}' for b in range(1, 17)] #Bands you want to get data from

#Change depending on what you want directory to be called
lat0, lon0 = 34.05, -118.2 #Location you want to sample from (will be in the centre of time series patch)
download_dir = 'satellite_data_la2'
series_dir = 'time_series_la2'
os.makedirs(download_dir, exist_ok=True)
os.makedirs(series_dir, exist_ok=True)


km_width = 100
fs = s3fs.S3FileSystem(anon=True)

#Find the 100 x 100 km width patch of the desired location. 
def get_pixel_window(ds, lat0, lon0, km_width):
    proj_info = ds['goes_imager_projection'].attrs
    h = proj_info['perspective_point_height']
    sweep = proj_info['sweep_angle_axis']
    lon_0 = proj_info['longitude_of_projection_origin']

    crs = CRS.from_proj4(f"+proj=geos +h={h} +lon_0={lon_0} +sweep={sweep} +ellps=GRS80 +units=m +no_defs")
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x_center, y_center = transformer.transform(lon0, lat0)

    x = ds['x'].values * h
    y = ds['y'].values * h
    dx = abs(x[1] - x[0])
    dy = abs(y[1] - y[0])

    n_px_x = round((km_width * 1000) / dx / 2)
    n_px_y = round((km_width * 1000) / dy / 2)

    x_idx = np.argmin(abs(x - x_center))
    y_idx = np.argmin(abs(y - y_center))

    x0 = max(0, x_idx - n_px_x)
    x1 = min(len(x), x_idx + n_px_x)
    y0 = max(0, y_idx - n_px_y)
    y1 = min(len(y), y_idx + n_px_y)

    return y0, y1, x0, x1

# Download and crop the frame.
def download_and_crop_file(s3_path):
    file_name = s3_path.split('/')[-1]
    band_code = file_name.split('_')[3][-4:]
    local_path = os.path.join(download_dir, f'cropped_{band_code}_{file_name}')
    if os.path.exists(local_path):
        return

    try:
        with fs.open(s3_path, 'rb') as f:
            ds = xr.open_dataset(f, engine='h5netcdf') #make sure the engine is correct
            if 'time' in ds.dims:
                ds = ds.isel(time=0)

            y0, y1, x0, x1 = get_pixel_window(ds, lat0, lon0, km_width)
            cropped = ds.isel(x=slice(x0, x1), y=slice(y0, y1)).load()

	# you need to convert coordidnates
            cropped = cropped.assign_coords( x=cropped.coords['x'].astype('float32'),y=cropped.coords['y'].astype('float32'))
            cropped['x'].attrs['_FillValue'] = np.nan
            cropped['y'].attrs['_FillValue'] = np.nan

            for coord in ['x', 'y', 'time']:
                if coord in cropped.coords:
                    cropped[coord].encoding = {}

            for var in cropped.variables:
                if 'units' in cropped[var].attrs:
                    del cropped[var].attrs['units']

            if 'time' in cropped.coords:
                cropped['time'].encoding = {'dtype': 'float64'}

            encoding = {var: {"zlib": True, "complevel": 1} for var in cropped.data_vars}
            cropped.to_netcdf(local_path, encoding=encoding)

    except Exception as e:
        print(f"Failed to crop {file_name}: {e}")

# Run function
if __name__ == "__main__":
    download_tasks = []
    for i in tqdm(range(num_days), desc="Collecting file paths"): #Collect file paths to save time
        date = start_date + timedelta(days=i)
        year = date.year
        doy = date.timetuple().tm_yday
        for hour in range(24):
            prefix = f'{path_name}/{year}/{doy:03}/{hour:02}/'
            try:
                files = fs.ls(prefix)
                band_files = [f for f in files if any(b in f for b in bands)]
                download_tasks.extend(band_files)
            except FileNotFoundError:
                continue

    print(f"Downloading and cropping {len(download_tasks)} files")

    with get_context("spawn").Pool(processes=16) as pool: #have more workers to speed things up.
        futures = [pool.apply_async(download_and_crop_file, (path,)) for path in download_tasks]
        for f in tqdm(futures, total=len(futures), desc="Downloading"):
            f.get()

    print("All downloads and crops completed.")

  