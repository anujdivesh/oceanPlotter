
import os, sys
current_directory = os.path.dirname(os.path.abspath(__file__))
tasks_code_path = os.path.join(current_directory, 'code')
sys.path.append(tasks_code_path)
from controller_task import initialize_taskController
from datetime import datetime
from utility_functions import Utility
from controller_server_path import PathManager
from update_thredds import thredds
import xml.etree.ElementTree as xmltree
from owslib.wms import WebMapService
import pandas as pd
from controller_server_path import PathManager
from datetime import datetime, timedelta
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta

def get_specific_stamp_bureau(data):
    #wms = WebMapService(data['url'], version="1.3.0")
    #layer = data['layer_name']
    #time = wms[layer].dimensions['time']['values']
    url = data['url']
    new_text = url.replace("wms", "dodsC")
    ds = xr.open_dataset(new_text)
    values = ds['time'].values  # This retrieves the raw time values

    dates = values.astype('datetime64[ms]').tolist()


    #dates = [datetime.strptime(x.decode('utf-8'), '%Y-%m-%dT%H:%M:%SZ') for x in values]

    # Convert datetime objects to the desired format '%Y-%m-%dT%H:%M:%S'
    formatted_dates = [dt.strftime('%Y-%m-%dT%H:%M:%SZ') for dt in dates]

    return formatted_dates

def get_specific_stamp(data):
    wms = WebMapService(data['url'], version="1.3.0")
    layer = data['layer_name']
    time = wms[layer].dimensions['time']['values']
    url = data['url']
    new_text = url.replace("wms", "dodsC")
    ds = xr.open_dataset(new_text)
    values = ds['time'].values  # This retrieves the raw time values

    dates = [datetime.strptime(x.decode('utf-8'), '%Y-%m-%dT%H:%M:%SZ') for x in values]

    # Convert datetime objects to the desired format '%Y-%m-%dT%H:%M:%S'
    formatted_dates = [dt.strftime('%Y-%m-%dT%H:%M:%SZ') for dt in dates]

    return formatted_dates


def generate_daily_dates(start_str: str, end_str: str, date_format: str = "%Y-%m-%dT%H:%M:%S.000Z"):

    start_time = datetime.strptime(start_str, date_format)
    end_time = datetime.strptime(end_str, date_format)

    date_list = []
    current_date = start_time

    while current_date <= end_time:
        formatted_date = current_date.strftime("%Y-%m-%dT%H:%M:%SZ")
        date_list.append(formatted_date)
        current_date += timedelta(days=1)

    return date_list


def get_specific(data):
    wms = WebMapService(data['url'], version="1.3.0")
    layer = data['layer_name']
    time = wms[layer].dimensions['time']['values']
    start_time = thredds.process_string_2(time[0])
    end_time = thredds.process_string_3(time[-1])
    return start_time,end_time

def generate_6_hour_intervals(start_str: str, end_str: str, date_format: str = "%Y-%m-%dT%H:%M:%S.000Z"):
    start_time = datetime.strptime(start_str, date_format)
    end_time = datetime.strptime(end_str, date_format)

    interval_hours = 6  # 6-hour intervals
    intervals = []

    current_time = start_time
    while current_time <= end_time:
        formatted_time = current_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        intervals.append(formatted_time)
        current_time += timedelta(hours=interval_hours)

    return intervals

def get_6hourly(data):
    wms = WebMapService(data['url'], version="1.3.0")
    layer = thredds.process_string(data['layer_name'])
    time = wms[layer].dimensions['time']['values'][0]
    start_time, end_time, period = time.split('/')
    return start_time,end_time

##COMMA PLOTS
comma_plots = [15,4,5,3,28,29,30,17]

for x in comma_plots:
    api_url = PathManager.get_url('ocean-api',"layer_web_map/"+str(x)+"/")
    api_response = thredds.get_data_from_api(api_url)
    start_time,end_time = get_specific(api_response)
    dates = generate_daily_dates(start_time,end_time)
    #last_10_dates = dates[-10:]
    for date in dates:
        print(date)
        #TOUCH THE API HERE

opendap_plots = [8,35,36,37,9,6,26,31,32,38]

for x in opendap_plots:
    api_url = PathManager.get_url('ocean-api',"layer_web_map/"+str(x)+"/")
    api_response = thredds.get_data_from_api(api_url)
    dates = get_specific_stamp(api_response)
    for date in dates:
        print(date)
        #TOUCH THE API HERE

bom_plots = [19,6,16,18]
for x in bom_plots:
    api_url = PathManager.get_url('ocean-api',"layer_web_map/"+str(x)+"/")
    api_response = thredds.get_data_from_api(api_url)
    dates = get_specific_stamp_bureau(api_response)
    for date in dates:
        print(date)

sixHourly_plot = [2,11,13,10,14,12,27]
for x in sixHourly_plot:
    api_url = PathManager.get_url('ocean-api',"layer_web_map/"+str(x)+"/")
    api_response = thredds.get_data_from_api(api_url)
    start_date, end_date = get_6hourly(api_response)
    print(start_date, end_date)
    dates = generate_6_hour_intervals(start_date,end_date)
    for date in dates:
        print(date)
        #TOUCH THE API HERE