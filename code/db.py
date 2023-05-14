from datetime import datetime
import pandas as pd
import requests
import psycopg2
from psycopg2.extensions import AsIs
from scrapeSolarData import scrape_solar_data
import csv

# ,user="leoposc",password="4xhm!zMY@dAVNQ7"
# database="weatherData"

def get_solar_data():
    return None


def connect(func):
    def wrapper(*args, **kwargs):
        # connect to weather data base
        conn = psycopg2.connect(database="bachelor")
        # open a cursor
        cur = conn.cursor()
        # call function
        # execute task and commit changes to db 
        values = func(cur, *args, **kwargs)
        conn.commit()
        # disconnect
        conn.close()
        cur.close()
        return values
        
    return wrapper
    

class DBManager():

    @connect
    def create_table(cur, self, location: str,table_type: str, solarsystem_id=None):

        assert(len(location) < 30)

        if table_type=="WEATHER":
            name = ("weatherdata_"+location.strip()).lower()

            cmd = f"""CREATE TABLE IF NOT EXISTS {name} ( 
            timeepoch INT PRIMARY KEY,
            hour SMALLINT,
            calendarweek SMALLINT,
            solarradiation SMALLINT,
            temperature SMALLINT,
            cloudcoverage SMALLINT,
            humidity SMALLINT,
            wind SMALLINT
            );"""

        elif table_type=="SOLAR":

            name = ("solardata_"+location.strip()+"_"+str(solarsystem_id)).lower()

            cmd = f"""CREATE TABLE IF NOT EXISTS {name} (
            timeepoch INT PRIMARY KEY,
            energyoutput INT);"""

        cur.execute(cmd)


    @connect
    def fetch_solar_data(cur, self, start: str, end: str, solarsystem_id: int):

        solar_data, location = scrape_solar_data(start, end, solarsystem_id)

        table_name = ("solardata_"+location.strip()+"_"+str(solarsystem_id)).lower()

        # find existing tables in database
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        tables = cur.fetchall()

        # check if table already exists
        if (table_name,) not in tables:
            # create table
            self.create_table(location, "SOLAR", solarsystem_id)        

        for row in solar_data:

            if not row:
                continue

            # insert row into db
            cur.execute("""
                INSERT INTO %s (timeepoch, energyoutput)
                VALUES (%s, %s)
            """, (AsIs(table_name), row[0], row[1]))


    @connect
    def fetch_weather_data(cur, self,location: str,start: str, end: str):

        table_name = ("weatherdata_"+location.strip()).lower()

        # find existing tables in database
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        tables = cur.fetchall()

        # check if table already exists
        if (table_name,) not in tables:
            # create table
            self.create_table(location, "WEATHER")

        # sort the table by timeEpoch
        cur.execute("""
            SELECT timeepoch FROM %s ORDER BY timeepoch ASC
        """, (AsIs(table_name),))
        sorted_table = cur.fetchall()

        if len(sorted_table) != 0:
            start_timestamp_database = sorted_table[0][0]
            end_timestamp_database   = sorted_table[-1][0]

            # convert start to timestamp
            start_timestamp_requested = int(datetime.strptime(start, '%Y-%m-%d').timestamp())
            # convert end to timestamp from this day at 23:00:00
            end_timestamp_requested = int(datetime.strptime(end, '%Y-%m-%d').replace(hour=23, minute=0, second=0).timestamp())

            # check if the requested time range is already in the database
            if start_timestamp_requested >= start_timestamp_database and end_timestamp_requested <= end_timestamp_database:
                print("The requested time range is already in the database.")
                return None
            
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

        key = "4MZDYZUR9MG5MTY4K8WJVT8K6"

        url = f'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{start}/{end}?unitGroup=metric&elements=datetime%2CdatetimeEpoch%2Ctemp%2Chumidity%2Cwindspeed%2Ccloudcover%2Csolarradiation%2Csolarenergy%2Cuvindex&include=hours%2Cobs%2Cremote&key={key}&contentType=csv'
                
        result = requests.get(url)

        csv_data = result.text.split('\n')
        csv_reader = csv.reader(csv_data, delimiter=',')
        csv_header = next(csv_reader)
                                                                                        #  csv_header = ['datetime',
                                                                                        #  'temp',
                                                                                        #  'humidity',
                                                                                        #  'windspeed',
                                                                                        #  'sealevelpressure',
                                                                                        #  'cloudcover',
                                                                                        #  'solarradiation',
                                                                                        #  'solarenergy',
                                                                                        #  'uvindex']
        print(csv_header)
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
            for i in range(1, len(row)-2):
                if row[i] == '':
                    # change missing values to None or 0 if it is the solar radiation
                    insert_row[i+2] = None if i != len(row)-2 else 0
                else:
                    insert_row[i+2] = int(float(row[i]))

            print("Row:        ", row)
            print("Insert_row: ", insert_row)

            # insert row into db    
            # cur.execute("""
            #     INSERT INTO %s (timeepoch, 
            #     hour, calendarweek, temperature, humidity, wind,
            #     cloudcoverage, solarradiation)
            #     VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            # """, (AsIs(table_name), *insert_row))


    @connect
    def select_solar_data(cur, self, location: str, solarsystem_id: str):
        
        table_name = ("solardata_"+location.strip()+"_"+str(solarsystem_id)).lower()

        # check if table even exists
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        tables = cur.fetchall()

        assert (table_name,) in tables, f"The table '{table_name}' does not exist."

        # get data from db
        cur.execute("""
            SELECT * FROM %s
        """, (AsIs(table_name),))

        solar_data = cur.fetchall()

        #convert to pandas dataframe
        df = pd.DataFrame(solar_data, columns=['timeepoch', 'energyoutput'])
        return df
    

    @connect
    def select_weather_data(cur, self, location: str):

        table_name = ('weatherdata_'+location.strip()).lower()

        # check if table even exists
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        tables = cur.fetchall()        

        assert (table_name,) in tables, f"The table '{table_name}' does not exist."

        # get data from db
        cur.execute("""
            SELECT * FROM %s
        """, (AsIs(table_name),))

        weather_data = cur.fetchall()

        #convert to pandas dataframe
        df = pd.DataFrame(weather_data, columns=['timeepoch', 'hour', 'calendarweek', 'temperature', 'humidity', 'wind', 'cloudcoverage', 'solarradiation'])
        return df


# manager = DBManager()
# manager.create_table('Stuttgart',"WEATHER")
# manager.select_solar_data('Stuttgart', '1')
# manager.select_weather_data('Stuttgart')
# manager.fetch_weather_data('Stuttgart','2023-02-01','2023-02-01')

manager = DBManager()
# manager.fetch_solar_data("2022-01-05", "2022-01-06", 10)
# print(manager.select_solar_data("applewood", "10"))s
manager.fetch_weather_data("applewood", "2022-01-05", "2022-01-06")
# print(manager.select_weather_data("applewood"))