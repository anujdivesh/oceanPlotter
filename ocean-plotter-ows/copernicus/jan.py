import os
import shutil
from datetime import datetime, timedelta
from copernicusmarine import subset

download_directory = "/Users/anujdivesh/Desktop/django/plotter/ows_plots"
dataset_id = "cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.125deg_P1D"
varibales = ["adt", "err_sla", "err_ugosa", "err_vgosa", "flag_ice", "sla", "ugos", "ugosa", "vgos", "vgosa"]
minimum_longitude= 100
maximum_longitude= 300
minimum_latitude= -45
maximum_latitude= 45

# Start and end dates for the loop
start_date = datetime(2025, 1, 1)  # January 1, 2025
end_date = datetime(2025, 1, 31)   # January 31, 2025

current_date = start_date

while current_date <= end_date:
    # Format the current date for the filename and request
    start_datetime = current_date.strftime("%Y-%m-%d %H:%M:%S")
    end_datetime = start_datetime  # Same for start and end for daily data

    date_str = current_date.strftime("%Y%m%d_%Y%m%d")
    new_file_name = f"nrt_global_allsat_phy_l4_{date_str}.nc"

    # Download the data for the current date
    subset(
            dataset_id=dataset_id,
            service='arco-geo-series',
            variables=varibales,
            minimum_longitude=minimum_longitude,
            maximum_longitude=maximum_longitude,
            minimum_latitude=minimum_latitude,
            maximum_latitude=maximum_latitude,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            force_download=True,
            overwrite_output_data=True,
            output_filename=new_file_name,
            output_directory=download_directory,
            credentials_file="/Users/anujdivesh/Desktop/django/plotter/ows_plots/.copernicusmarine-credentials"
    )

    # Move to the next day
    current_date += timedelta(days=1)
