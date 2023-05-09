#%%
import psycopg2
import requests
import csv
from datetime import datetime

conn = psycopg2.connect(database="bachelor")

# open a cursor
cur = conn.cursor()

#%%
start = '2023-02-1'
end = '2023-05-01'

# sort the table by timeEpoch
cur.execute("""
    SELECT timeEpoch FROM weatherData_Stuttgart ORDER BY timeEpoch ASC
""")
sorted_table = cur.fetchall()
print(sorted_table)
start_timestamp_database = sorted_table[0][0]
end_timestamp_database   = sorted_table[-1][0]

# convert start to timestamp
start_timestamp_requested = int(datetime.strptime(start, '%Y-%m-%d').timestamp())
# convert end to timestamp from this day at 23:00:00
end_timestamp_requested = int(datetime.strptime(end, '%Y-%m-%d').replace(hour=23, minute=0, second=0).timestamp())

# check if the requested time range is already in the database
if start_timestamp_requested >= start_timestamp_database and end_timestamp_requested <= end_timestamp_database:
    print("The requested time range is already in the database.")
# check if the ending of the requested time range is in the database, but not the beginning
elif start_timestamp_requested < start_timestamp_database and end_timestamp_requested < end_timestamp_database:
    # set end_timestamp_requested to one day before the start_timestamp_database
    end_timestamp_requested = start_timestamp_database - 86400
    # set end to one day before the start
    end = datetime.fromtimestamp(end_timestamp_requested).strftime('%Y-%m-%d')
    print("The ending of the requested time range is already in the database.")
#check if beginning of requested time range is in the database, but not the end
elif start_timestamp_requested > start_timestamp_database and end_timestamp_requested > end_timestamp_database:
    # set start_timestamp_requested to one day after the end_timestamp_database
    start_timestamp_requested = end_timestamp_database + 86400
    # set start to one day after the end
    start = datetime.fromtimestamp(start_timestamp_requested).strftime('%Y-%m-%d')
    print("The beginning of the requested time range is already in the database.")

#%%
# function: fetch weather data

key = "4MZDYZUR9MG5MTY4K8WJVT8K6"
location = 'Stuttgart'

# url = f'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{start}/{end}?unitGroup=metric&elements=datetime%2CdatetimeEpoch%2Ctemp%2Chumidity%2Cwindspeed%2Ccloudcover%2Csolarradiation%2Csolarenergy%2Cuvindex&include=hours%2Cobs%2Cremote&key={key}&contentType=csv'
url = f'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{start}/{end}?unitGroup=metric&elements=datetime%2CdatetimeEpoch%2Ctemp%2Chumidity%2Cwindspeed%2Ccloudcover%2Csolarradiation&include=hours%2Cobs%2Cremote&key={key}&contentType=csv'     

result = requests.get(url)

csv_data = result.text.split('\n')

#%%
# store csv_reader in db
csv_reader = csv.reader(csv_data, delimiter=',')
csv_header = next(csv_reader)
# csv_header = ['datetime',
#  'temp',
#  'humidity',
#  'windspeed',
#  'sealevelpressure',
#  'cloudcover',
#  'solarradiation',
#  'solarenergy',
#  'uvindex']

for row in csv_reader:

    if not row:        
        continue

    insert_row = [None] * 8
    # convert 2023-02-01T00:00:00 to timestamp
    insert_row[0] = datetime.strptime(row[0], '%Y-%m-%dT%H:%M:%S').timestamp()
    # get hour from timestamp
    insert_row[1] = datetime.fromtimestamp(insert_row[0]).hour
    # get calendar week from timestamp
    insert_row[2] = datetime.fromtimestamp(insert_row[0]).isocalendar()[1]
    for i in range(1, len(row)):
        if row[i] == '':
            # change missing values to None or 0 if it is the solar radiation
            insert_row[i+2] = None if i != len(row)-1 else 0
        else:
            insert_row[i+2] = int(float(row[i]))

    # insert row into db    
    cur.execute("""
        INSERT INTO weatherData_Stuttgart (timeEpoch, 
        hour, calendarWeek, temperature, humidity, wind,
        cloudCoverage, solarRadiation)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """, insert_row)

    


#%%
conn.commit()
# disconnect
conn.close()
cur.close()       

# %%
