import requests
import csv
from bs4 import BeautifulSoup
from statistics import mean
from datetime import datetime
from geopy.geocoders import Nominatim


def scrape_solar_data(start: str, end: str, id: int):
    # initialize geolocator
    geolocation = Nominatim(user_agent="leopldSchmid")
    # Set the year you want to download data for
    # calculate the number of days between start and end
    days = int((datetime.strptime(end, '%Y-%m-%d') - datetime.strptime(start, '%Y-%m-%d')).days) + 1
    records_total = days * 24
    start = start.split('-')
    end = end.split('-')
    year_start = int(start[0])
    month_start = int(start[1])
    day_start = int(start[2])
    year_end = int(end[0])
    month_end = int(end[1])
    day_end = int(end[2])

    # Set the number of objects to download data for
    # system_ids = [10,1199,1200,1201,1202,1203,1204,1207,1209]

    # find the location of the solar systems
    url = 'https://oedi-data-lake.s3.amazonaws.com/pvdaq/csv/systems.csv'

    # Make the website request and parse the response using BeautifulSoup
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    lines = str(soup).strip().split("\n")

    # get the city name of the solar system
    for line in lines[1:]:
        # split elements of the row by comma
        values = line.split(",")        
        system_id = values[0].strip('"')

        if int(system_id) != id:
            continue

        latitude = (values[5].strip('"'))
        longitude = (values[6].strip('"'))
        location = geolocation.reverse(f"{latitude}, {longitude}")
        # get city name
        try:
            city = location.raw['address']['city']
        except KeyError:
            try:
                city = location.raw['address']['town']
            except KeyError:
                city = location.raw['address'].get('village', 'Unknown')

    # initialize list for solar data
    solar_data = [[0, 0] for i in range(records_total)]
    data_idx = 0 
    
    # loop through each year
    for year in range(year_start, year_end+1):
        
        #loop through each month
        for month in range(month_start, month_end+1):
            url = f"https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=pvdaq%2Fcsv%2Fpvdata%2Fsystem_id%3D{id}%2Fyear%3D{year}%2Fmonth%3D{month}%2F"

            # Make the website request and parse the response using BeautifulSoup
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            try:
                # find the amount of entries for the month. Look at class="dataTables_info" and split the text by spaces. The second to last word is the number of entries
                entries = soup.find("div", class_="dataTables_info").text.split()[-2]
            except AttributeError:
                raise Exception(f"Could not find any data for system {id} in {year}-{month:02d}")
            
            for entry in range(day_start, min(int(entries), day_end+1)):
                csv_link = f"https://oedi-data-lake.s3.amazonaws.com/pvdaq/csv/pvdata/system_id={id}/year={year}/month={month}/day={entry}/system_{id}__date_{year}_{month:02d}_{entry:02d}.csv"
                response = requests.get(csv_link)
                data = response.content.decode("utf-8")
                csvreader = csv.reader(data.splitlines())

                # Skip the headers row
                header = next(csvreader)
                TARGET_INDEX = 2
                # assert header[9] == "dc_power__422", "dc_power__422 is not in the 10th column"
                assert header[TARGET_INDEX] == "ac_power_metered_kw__4197", "ac_power_metered_kw__4197 is not in the 2nd column"

                # Write the data rows to the CSV file
                # loop through every sixty rows and get the mean value for each hour
                data = list(csvreader)
                # calculate time interval between each entry (in minutes)
                minutes_passed_till_second_entry = datetime.strptime(data[1][0], '%Y-%m-%d %H:%M:%S').minute
                time_interval = 60 // minutes_passed_till_second_entry
                for idx in range(0,len(data),time_interval):                  
                    solar_data[data_idx][0] = int(datetime.strptime(data[idx][0], '%Y-%m-%d %H:%M:%S').timestamp())
                    val =  mean([int(float(x[TARGET_INDEX])) for x in data[idx:idx+time_interval]])
                    solar_data[data_idx][1] = val if val > 30 else 0
                    data_idx += 1
                
    return solar_data, city


# print(scrape_solar_data('2022-01-01', '2022-01-10', 10))