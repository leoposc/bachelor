import requests
import csv
from bs4 import BeautifulSoup

# Set the year you want to download data for
year = "2022"

# Set the number of objects to download data for
# system_ids = [10,1199,1200,1201,1202,1203,1204,1207,1209]
system_ids = [10]

# # Set the URL for the website
# url = f"https://data.openei.org/s3_viewer?bucket=oedi-data-lake&prefix=pvdaq%2Fcsv%2Fpvdata%2F"

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

            # find the amount of entries for the month. Look at "Showing 1 to X of X entries"
            print(soup)
            # entries = soup.find("div", class_="dataTables_info").text.split()[-2] 

            # # Find the link to the CSV file and download the data
            # csv_link = soup.find("a", text=f"{year}_data.csv")["href"]
            # response = requests.get(csv_link)
            # data = response.content.decode("utf-8")
            # csvreader = csv.reader(data.splitlines())

            # # Skip the headers row
            # next(csvreader)

            # # Write the data rows to the CSV file
            # for row in csvreader:
            #     csvwriter.writerow(row)
