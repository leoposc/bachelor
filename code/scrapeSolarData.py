#%%
import requests
import csv
from bs4 import BeautifulSoup
from statistics import mean
from datetime import datetime
from geopy.geocoders import Nominatim

# initialize geolocator
geolocation = Nominatim(user_agent="leopldSchmid")

# Set the year you want to download data for
year = "2022"

# Set the number of objects to download data for
# system_ids = [10,1199,1200,1201,1202,1203,1204,1207,1209]
system_ids = [10]

# find the location of the solar systems
url = 'https://oedi-data-lake.s3.amazonaws.com/pvdaq/csv/systems.csv'

# Make the website request and parse the response using BeautifulSoup
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")

systems_metadata = dict()

lines = str(soup).strip().split("\n")
print(lines[0])

for line in lines:
    # split elements of the row by comma
    values = line.split(",")
    
    system_id = values[0].strip('"')
    print(system_id)
    print(system_id not in system_ids)

    if system_id not in system_ids:
        continue

    latitude = values[5].strip('"')
    longitude = values[6].strip('"')
    location = geolocation.reverse(f"{latitude, longitude}")
    systems_metadata[system_id] = location.raw['address']['city']




#%%


for system_id in system_ids:
    # Create a CSV file to write the data to
    filename = f"pvdata_{system_id}_{year}.csv"
    with open(filename, mode="w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=",")

        # Write the headers to the CSV file
        csvwriter.writerow(["Timestamp", "Global Horizontal Irradiance (W/m²)", "Direct Normal Irradiance (W/m²)", "Diffuse Horizontal Irradiance (W/m²)", "Air Temperature (C)", "Wind Speed (m/s)", "Relative Humidity (%)", "Ac Power (W)"])

        #loop through each month
        for month in range(1,2):
            url = f"https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=pvdaq%2Fcsv%2Fpvdata%2Fsystem_id%3D{system_id}%2Fyear%3D{year}%2Fmonth%3D{month}%2F"

            # Make the website request and parse the response using BeautifulSoup
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")

            # find the amount of entries for the month. Look at class="dataTables_info" and split the text by spaces. The second to last word is the number of entries
            entries = soup.find("div", class_="dataTables_info").text.split()[-2]
            # print(soup)
            
            for entry in range(1, 2):#int(entries)):
                csv_link = f"https://oedi-data-lake.s3.amazonaws.com/pvdaq/csv/pvdata/system_id={system_id}/year={year}/month={month}/day={entry}/system_{system_id}__date_{year}_{month:02d}_{entry:02d}.csv"
                response = requests.get(csv_link)
                data = response.content.decode("utf-8")
                csvreader = csv.reader(data.splitlines())

                # Skip the headers row
                header = next(csvreader)
                assert header[9] == "dc_power__422", "dc_power__422 is not in the 10th column"

                # Write the data rows to the CSV file
                # loop through every sixty rows and get the mean value for each hour
                data = list(csvreader)
                for idx in range(0,len(data),60):
                    insert_row = [0,0]
                    time = data[idx][0]
                    # convert '2022-01-29 23:00:00' to unix timestamp
                    insert_row[0] = int(datetime.strptime(data[idx][0], '%Y-%m-%d %H:%M:%S').timestamp())

                    val =  mean([int(float(x[9])) for x in data[idx:idx+60]])
                    insert_row[1] = val if val > 0 else 0
                    print(insert_row)

                    csvwriter.writerow(insert_row)
                    

#%%




