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
today = datetime.now()
yesterday = today - timedelta(days=3)
#start_datetime = yesterday.strftime("%Y-%m-%dT%H:%M:%S")
#end_datetime = start_datetime
start_datetime = "2025-01-01 00:00:00"
end_datetime = "2025-01-01 00:00:00"

date_str = yesterday.strftime("%Y%m%d_%Y%m%d")
new_file_name = f"nrt_global_allsat_phy_l4_{date_str}.nc"
subset(
        dataset_id=dataset_id,
        service = 'arco-geo-series',
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

"""
# Define the directory where you want to save the downloaded data
download_directory = "/home/pop/Desktop/ocean-portal2.0/backend_design"
 
# Create the directory if it doesn't exist
os.makedirs(download_directory, exist_ok=True)
 
# Specify the dataset ID and version
dataset_id = "cmems_obs-sl_glo_phy-ssh_nrt_allsat-l4-duacs-0.25deg_P1D"
dataset_version = "202311"
 
# Define the geographical and temporal subset parameters
parameters = {
    "variables": ["adt", "err_sla", "err_ugosa", "err_vgosa", "flag_ice", "sla", "ugos", "ugosa", "vgos", "vgosa"],
    "minimum_longitude": 100,
    "maximum_longitude": 300,
    "minimum_latitude": -45,
    "maximum_latitude": 45,
    "force_download": True,
    "overwrite_output_data":True,
    "credentials_file":""

}
today = datetime.now()
# Function to check data availability and download if available
def check_and_download():
    for i in range(6):  # range(6) to include today and the past 5 days
        yesterday = today - timedelta(days=i)
        print(yesterday)
        start_datetime = yesterday.strftime("%Y-%m-%dT%H:%M:%S")
        end_datetime = start_datetime  # Same for both start and end as we want data for the specific datetime
        date_str = yesterday.strftime("%Y%m%d_%Y%m%d")
        new_file_name = f"nrt_global_allsat_phy_l4_{date_str}.nc"
        month =  yesterday.strftime('%m')
        year = yesterday.strftime('%Y')
        print(month)
        # Update the time parameters
        parameters.update({
            "start_datetime": start_datetime,
            "end_datetime": end_datetime,
            "output_filename" : new_file_name,
            "output_directory" : download_directory
        })
    
        print(f"Checking for data from {start_datetime} to {end_datetime}")
    
        try:
            # Use the subset function to download the data
            subset(
                dataset_id=dataset_id,
                dataset_version=dataset_version,
                **parameters
            )
            #os.system("scp "+download_directory+"/"+new_file_name+" ubuntu@13.211.201.138:/data/sea_level/grids/daily/"+year+"/"+month+"/"+new_file_name)
    
        except Exception as e:
            print(f"Error while checking or downloading data: {e}")
        
check_and_download()
"""