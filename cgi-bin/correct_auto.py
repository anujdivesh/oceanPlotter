def task_2():
    layer_id = [5, 7]

    for id in layer_id:
        api_url = PathManager.get_url('ocean-api', "layer_web_map/" + str(id) + "/")
        api_response = thredds.get_data_from_api(api_url)

        if api_response['period'] == "COMMA":
            thredds.get_specific(api_response)
            dates = thredds.generate_daily_dates(start_time, end_time)

            for date in dates:
                print(date)
                for y in countries:
                    url = "https://ocean-cgi.spc.int/cgi-bin/plot_opendap.py?region=%s&layer_map=%s&time=%s" % (y['id'], str(id), date)
                    response = requests.get(url)

                    if response.status_code == 200:
                        print(f"Success: Received 200 OK from {url}")
                    else:
                        print(f"Failed: Status code {response.status_code} from {url}")

                    print(y['id'])
        else:
            raise ValueError("Dataset Period not found.")


def task_2():
  layer_id = [6,38]

  for id in layer_id:
      api_url = PathManager.get_url('ocean-api',"layer_web_map/"+str(id)+"/")
      
      api_response = thredds.get_data_from_api(api_url)
      if api_response['period'] == "OPENDAP":
          thredds.get_specific_stamp_bureau(api_response)
          countries = thredds.get_country_data()
          dates = thredds.get_specific_stamp_bureau_update(api_response)
          for date in dates:
              print(date)
              for y in countries:
                  url = "https://ocean-cgi.spc.int/cgi-bin/plot_opendap.py?region=%s&layer_map=%s&time=%s" % (y['id'],str(id),date)
                  response = requests.get(url)
                  if response.status_code == 200:
                      print(f"Success: Received 200 OK from {url}")
                  else:
                      print(f"Failed: Status code {response.status_code} from {url}")
      else:
          raise ValueError("Dataset Period not found.")


def task_2():
  layer_id = [16]

  for id in layer_id:
      api_url = PathManager.get_url('ocean-api',"layer_web_map/"+str(id)+"/")
      
      api_response = thredds.get_data_from_api(api_url)
      if api_response['period'] == "OPENDAP":
          thredds.get_specific_stamp_bureau(api_response)
          countries = thredds.get_country_data()
          dates = thredds.get_specific_stamp_bureau_update(api_response)
          for date in dates:
              print(date)
              for y in countries:
                  url = "https://ocean-cgi.spc.int/cgi-bin/plot_opendap.py?region=%s&layer_map=%s&time=%s" % (y['id'],str(id),date)
                  response = requests.get(url)
                  if response.status_code == 200:
                      print(f"Success: Received 200 OK from {url}")
                  else:
                      print(f"Failed: Status code {response.status_code} from {url}")
      else:
          raise ValueError("Dataset Period not found.")


def task_2():
    layer_id = [17]
    for id in layer_id:
        api_url = PathManager.get_url('ocean-api', "layer_web_map/" + str(id) + "/")
        api_response = thredds.get_data_from_api(api_url)
        if api_response['period'] == "COMMA":
            thredds.get_specific(api_response)
            countries = thredds.get_country_data()
            start_time, end_time = thredds.get_specific_update(api_response)
            dates = thredds.generate_daily_dates(start_time, end_time)
            for date in dates:
                print(date)
                for y in countries:
                    url = "https://ocean-cgi.spc.int/cgi-bin/plot_opendap.py?region=%s&layer_map=%s&time=%s" % (y['id'], str(id), date)
                    response = requests.get(url)
                    if response.status_code == 200:
                        print(f"Success: Received 200 OK from {url}")
                    else:
                        print(f"Failed: Status code {response.status_code} from {url}")
                    print(y['id'])
        else:
            raise ValueError("Dataset Period not found.")


def task_22():
    layer_id = [26]
    for id in layer_id:
        api_url = PathManager.get_url('ocean-api', "layer_web_map/" + str(id) + "/")
        api_response = thredds.get_data_from_api(api_url)
        if api_response['period'] == "OPENDAP":
            thredds.get_specific_stamp(api_response)
            countries = thredds.get_country_data()
            dates = thredds.get_specific_stamp_update(api_response)
            for date in dates:
                print(date)
                for y in countries:
                    url = "https://ocean-cgi.spc.int/cgi-bin/plot_opendap.py?region=%s&layer_map=%s&time=%s" % (y['id'], str(id), date)
                    response = requests.get(url)
                    if response.status_code == 200:
                        print(f"Success: Received 200 OK from {url}")
                    else:
                        print(f"Failed: Status code {response.status_code} from {url}")
                    print(y['id'])
        else:
            raise ValueError("Dataset Period not found.")



def task_2():
  layer_id = [18]

  for id in layer_id:
      api_url = PathManager.get_url('ocean-api',"layer_web_map/"+str(id)+"/")
      
      api_response = thredds.get_data_from_api(api_url)
      if api_response['period'] == "OPENDAP":
          thredds.get_specific_stamp_bureau(api_response)
          countries = thredds.get_country_data()
          dates = thredds.get_specific_stamp_bureau_update(api_response)
          for date in dates:
              print(date)
              for y in countries:
                  url = "https://ocean-cgi.spc.int/cgi-bin/plot_opendap.py?region=%s&layer_map=%s&time=%s" % (y['id'],str(id),date)
                  response = requests.get(url)
                  if response.status_code == 200:
                      print(f"Success: Received 200 OK from {url}")
                  else:
                      print(f"Failed: Status code {response.status_code} from {url}")
      else:
          raise ValueError("Dataset Period not found.")


def task_2():
    layer_id = [2,11,13,10,14,12]

    for id in layer_id:
        api_url = PathManager.get_url('ocean-api',"layer_web_map/"+str(id)+"/")
        
        api_response = thredds.get_data_from_api(api_url)
        if api_response['period'] == "PT6H":
            thredds.get_6hourly(api_response)
            countries = thredds.get_country_data()
            start_date, end_date = thredds.get_6hourly_update(api_response)
            dates = thredds.generate_6_hour_intervals(start_date,end_date)
            for date in dates:
                for y in countries:
                    url = "https://ocean-cgi.spc.int/cgi-bin/plot_opendap.py?region=%s&layer_map=%s&time=%s" % (y['id'],str(x),date)
                    response = requests.get(url)
                    if response.status_code == 200:
                        print(f"Success: Received 200 OK from {url}")
                    else:
                        print(f"Failed: Status code {response.status_code} from {url}")

                    print(y['id'])
        else:
            raise ValueError("Dataset Period not found.")



def task_2():
    layer_id = [3]

    for id in layer_id:
        api_url = PathManager.get_url('ocean-api',"layer_web_map/"+str(id)+"/")
        
        api_response = thredds.get_data_from_api(api_url)
        if api_response['period'] == "COMMA":
            thredds.get_specific(api_response)
            countries = thredds.get_country_data()
            start_time, end_time = thredds.get_specific_update(api_response)
            dates = thredds.generate_daily_dates(start_time, end_time)
            for date in dates:
                print(date)
                for y in countries:
                    url = "https://ocean-cgi.spc.int/cgi-bin/plot_opendap.py?region=%s&layer_map=%s&time=%s" % (y['id'], str(id), date)
                    response = requests.get(url)
                    if response.status_code == 200:
                        print(f"Success: Received 200 OK from {url}")
                    else:
                        print(f"Failed: Status code {response.status_code} from {url}")
                    print(y['id'])
        else:
            raise ValueError("Dataset Period not found.")



def task_2():
    layer_id = [28]

    for id in layer_id:
        api_url = PathManager.get_url('ocean-api',"layer_web_map/"+str(id)+"/")
        
        api_response = thredds.get_data_from_api(api_url)
        if api_response['period'] == "COMMA":
            thredds.get_specific(api_response)
            countries = thredds.get_country_data()
            start_time, end_time = thredds.get_specific_update(api_response)
            dates = thredds.generate_daily_dates(start_time, end_time)
            for date in dates:
                print(date)
                for y in countries:
                    url = "https://ocean-cgi.spc.int/cgi-bin/plot_opendap.py?region=%s&layer_map=%s&time=%s" % (y['id'], str(id), date)
                    response = requests.get(url)
                    if response.status_code == 200:
                        print(f"Success: Received 200 OK from {url}")
                    else:
                        print(f"Failed: Status code {response.status_code} from {url}")
                    print(y['id'])
        else:
            raise ValueError("Dataset Period not found.")


def task_2():
    layer_id = [31]

    for id in layer_id:
        api_url = PathManager.get_url('ocean-api',"layer_web_map/"+str(id)+"/")
        
        api_response = thredds.get_data_from_api(api_url)
        if api_response['period'] == "OPENDAP":
            thredds.get_specific_stamp(api_response)
            countries = thredds.get_country_data()
            start_time, end_time = thredds.get_specific_update(api_response)
            dates = thredds.generate_daily_dates(start_time, end_time)
            for date in dates:
                print(date)
                for y in countries:
                    url = "https://ocean-cgi.spc.int/cgi-bin/plot_opendap.py?region=%s&layer_map=%s&time=%s" % (y['id'], str(id), date)
                    response = requests.get(url)
                    if response.status_code == 200:
                        print(f"Success: Received 200 OK from {url}")
                    else:
                        print(f"Failed: Status code {response.status_code} from {url}")
                    print(y['id'])
        else:
            raise ValueError("Dataset Period not found.")


def task_2():
    layer_id = [29]

    for id in layer_id:
        api_url = PathManager.get_url('ocean-api',"layer_web_map/"+str(id)+"/")
        
        api_response = thredds.get_data_from_api(api_url)
        if api_response['period'] == "COMMA":
            thredds.get_specific(api_response)
            countries = thredds.get_country_data()
            start_time, end_time = thredds.get_specific_update(api_response)
            dates = thredds.generate_daily_dates(start_time, end_time)
            for date in dates:
                print(date)
                for y in countries:
                    url = "https://ocean-cgi.spc.int/cgi-bin/plot_opendap.py?region=%s&layer_map=%s&time=%s" % (y['id'], str(id), date)
                    response = requests.get(url)
                    if response.status_code == 200:
                        print(f"Success: Received 200 OK from {url}")
                    else:
                        print(f"Failed: Status code {response.status_code} from {url}")
                    print(y['id'])
        else:
            raise ValueError("Dataset Period not found.")

def task_2():
    layer_id = [32]

    for id in layer_id:
        api_url = PathManager.get_url('ocean-api',"layer_web_map/"+str(id)+"/")
        
        api_response = thredds.get_data_from_api(api_url)
        if api_response['period'] == "OPENDAP":
            thredds.get_specific_stamp(api_response)
            countries = thredds.get_country_data()
            start_time, end_time = thredds.get_specific_update(api_response)
            dates = thredds.generate_daily_dates(start_time, end_time)
            for date in dates:
                print(date)
                for y in countries:
                    url = "https://ocean-cgi.spc.int/cgi-bin/plot_opendap.py?region=%s&layer_map=%s&time=%s" % (y['id'], str(id), date)
                    response = requests.get(url)
                    if response.status_code == 200:
                        print(f"Success: Received 200 OK from {url}")
                    else:
                        print(f"Failed: Status code {response.status_code} from {url}")
                    print(y['id'])
        else:
            raise ValueError("Dataset Period not found.")


def task_2():
    layer_id = [30]

    for id in layer_id:
        api_url = PathManager.get_url('ocean-api',"layer_web_map/"+str(id)+"/")
        
        api_response = thredds.get_data_from_api(api_url)
        if api_response['period'] == "COMMA":
            thredds.get_specific(api_response)
            countries = thredds.get_country_data()
            start_time, end_time = thredds.get_specific_update(api_response)
            dates = thredds.generate_daily_dates(start_time, end_time)
            for date in dates:
                print(date)
                for y in countries:
                    url = "https://ocean-cgi.spc.int/cgi-bin/plot_opendap.py?region=%s&layer_map=%s&time=%s" % (y['id'], str(id), date)
                    response = requests.get(url)
                    if response.status_code == 200:
                        print(f"Success: Received 200 OK from {url}")
                    else:
                        print(f"Failed: Status code {response.status_code} from {url}")
                    print(y['id'])
        else:
            raise ValueError("Dataset Period not found.")

def task_2():
    layer_id = [27]

    for id in layer_id:
        api_url = PathManager.get_url('ocean-api',"layer_web_map/"+str(id)+"/")
        
        api_response = thredds.get_data_from_api(api_url)
        if api_response['period'] == "PT6H":
            thredds.get_6hourly(api_response)
            countries = thredds.get_country_data()
            start_date, end_date = thredds.get_6hourly_update(api_response)
            dates = thredds.generate_6_hour_intervals(start_date,end_date)
            for date in dates:
                for y in countries:
                    url = "https://ocean-cgi.spc.int/cgi-bin/plot_opendap.py?region=%s&layer_map=%s&time=%s" % (y['id'],str(x),date)
                    response = requests.get(url)
                    if response.status_code == 200:
                        print(f"Success: Received 200 OK from {url}")
                    else:
                        print(f"Failed: Status code {response.status_code} from {url}")

                    print(y['id'])
        else:
            raise ValueError("Dataset Period not found.")