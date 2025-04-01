import requests

def fetch_wms_layer_data(api_url):
    # Make a GET request to fetch the data from the API
    response = requests.get(api_url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        
        # Extract the required values directly
        wms_url = data.get('url')
        layer_name = data.get('layer_name')
        transparent = data.get('transparent')
        styles = data.get('style')
        min_color = data.get('colormin')
        max_color = data.get('colormax')
        numcolorbands = data.get('numcolorbands')
        time_interval_start = data.get('timeIntervalStart')
        time_interval_end = data.get('timeIntervalEnd')
        period = data.get('period')
        interval_step = data.get('interval_step')
        logscale = data.get('logscale')
        
        # Return all extracted values
        return {
            'wms_url': wms_url,
            'layer_name': layer_name,
            'transparent': transparent,
            'styles': styles,
            'min_color': min_color,
            'max_color': max_color,
            'numcolorbands': numcolorbands,
            'time_interval_start': time_interval_start,
            'time_interval_end': time_interval_end,
            'period': period,
            'interval_step': interval_step,
            'logscale': logscale
        }
    else:
        # If request fails, return a message
        return {'error': f"Failed to fetch data. Status code: {response.status_code}"}

# Example usage
api_url = "https://dev-oceanportal.spc.int/middleware/api/layer_web_map/20/"
layer_data = fetch_wms_layer_data(api_url)

# Print the extracted data
print(layer_data['wms_url'])
