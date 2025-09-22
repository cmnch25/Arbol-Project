#Script to convert satellite data patches into 2D time series for each band. 

import os
import re
import xarray as xr
import numpy as np
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore", message=".*Times can't be serialized faithfully.*", category=UserWarning)


download_dir = 'satellite_data_la2' #Input directory
series_dir = 'time_series_la2' #output directory
os.makedirs(series_dir, exist_ok=True)

#get the time from the file name
def extract_time(fname):
    match = re.search(r"_s(\d{13})", fname)
    if match:
        return datetime.strptime(match.group(1), "%Y%j%H%M%S")
    return None


band_groups = defaultdict(list). # Group files by band 
for file in sorted(os.listdir(download_dir)):
    if file.endswith(".nc"):
        parts = file.split("_")
        if len(parts) < 4:
            continue
        band_code = parts[3][-3:]  # e.g., 'C01'
        band_groups[band_code].append(os.path.join(download_dir, file))

BATCH_SIZE = 100 #Group files in batches to speed things up 
print(" Generating time series files in batches.")

for band, files in tqdm(band_groups.items(), desc="Combining bands"):
    print(f"\n {band}: {len(files)} files total")

    valid_files = [f for f in files if os.path.getsize(f) > 0 and not os.path.basename(f).startswith("._")]   # Validate files
    print(f"{len(valid_files)} files validated for {band}")

    time_slices = []

    for i in range(0, len(valid_files), BATCH_SIZE): #Process files in each batch
        batch = valid_files[i:i+BATCH_SIZE]
        print(f" Loading batch {i//BATCH_SIZE + 1}/{(len(valid_files) + BATCH_SIZE - 1) // BATCH_SIZE} for {band} ({len(batch)}files).")

        batch_slices = []

        for file in tqdm(batch, desc=f"{band} batch files", leave=False): #Open file folder
            try:
                t = extract_time(file)
                if t is None:
                    continue

                ds = xr.open_dataset(file, engine="netcdf4")
                if 'crs' in ds.variables:
                    ds = ds.drop_vars('crs') #Drop unnecessary variables
              
                time_val = np.datetime64(t, 'ns') # Add time dimension and coordinate properly
                ds = ds.expand_dims(time=[time_val])
                ds = ds.assign_coords(time=("time", [time_val]))

                batch_slices.append(ds)

            except Exception as e:
                print(f" Skipping {file}: {e}")

        if not batch_slices:
            print(f"No valid datasets in batch {i//BATCH_SIZE + 1}, skipping.")
            continue

        try:
            ds_batch = xr.concat(batch_slices, dim='time')
            time_slices.append(ds_batch)
        except Exception as e:
            print(f"Failed to process batch {i//BATCH_SIZE + 1}: {e}")
            continue

    if not time_slices:
        print(f"Skipping band {band}: no usable data.")
        continue

    try:
        print(f" Concatenating all batches for {band}.")
        ds_all = xr.concat(time_slices, dim='time') # Join patches into time series
        ds_all = ds_all.sortby("time")

        # Encoding setup #Make sure everything has a value. 
        encoding = {'x': {'dtype': 'float32', '_FillValue': -9999.0},'y': {'dtype': 'float32', '_FillValue': -9999.0},'time':{'dtype': 'int64', 'units': 'seconds since 2025-03-01T00:00:00'}}
        for var in ds_all.data_vars:
            encoding[var] = {'zlib': True, 'complevel': 1}

        out_path = os.path.join(series_dir, f"{band}_time_series.nc")
        print(f" Saving {out_path}")
        ds_all.to_netcdf(out_path, encoding=encoding)
        print(f" Saved {out_path}")

    except Exception as e:
        print(f" Final concatenation failed for {band}: {e}")
