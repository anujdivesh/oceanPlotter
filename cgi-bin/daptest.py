import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
import numpy as np
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Open the dataset from the OpenDAP URL
url = "https://ocean-thredds.spc.int/thredds/dodsC/POP/model/regional/noaa/nrt/daily/sst_anomalies/latest.ncml"

try:
    ds = xr.open_dataset(url)
    print("Dataset opened successfully.")
    print(f"Available variables: {list(ds.variables.keys())}")
    print(f"Time variable type: {type(ds.time.values[0])}")
except Exception as e:
    print(f"Error opening dataset: {e}")
    exit()

# Specify the timestamp to plot
target_time = "2025-03-20T00:00:00Z"

try:
    # Convert bytes time to datetime
    time_str = [t.decode('utf-8') for t in ds.time.values]
    time_dt = np.array([datetime.strptime(t, "%Y-%m-%dT%H:%M:%SZ") for t in time_str])
    
    # Find the closest time to the target
    target_dt = datetime.strptime(target_time, "%Y-%m-%dT%H:%M:%SZ")
    time_index = abs(time_dt - target_dt).argmin()
    selected_time = time_str[time_index]
    
    print(f"\nTarget time: {target_time}")
    print(f"Closest available time in dataset: {selected_time}")

    # Extract SST anomaly data
    sst_anomaly = ds['anom'].isel(time=time_index)  # Note: using 'anom' instead of 'sst_anomaly'
    
    # Plotting
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Plot SST anomaly
    im = sst_anomaly.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='coolwarm',
        vmin=-3,  # Adjust based on data range
        vmax=3,   # Adjust based on data range
        add_colorbar=False
    )
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02, aspect=30)
    cbar.set_label('Sea Surface Temperature Anomaly (°C)')
    
    # Add map features
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.gridlines(draw_labels=True)
    
    # Title with the selected time
    plt.title(f'SST Anomalies on {selected_time}')
    
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"\nError processing data: {e}")
    if 'ds' in locals():
        print("\nAvailable times in dataset:")
        print(time_str)