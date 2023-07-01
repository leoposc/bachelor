from datetime import datetime
import pandas as pd
import requests
import psycopg2
from psycopg2.extensions import AsIs
from scrapeSolarData import scrape_solar_data
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
import csv

# ,user="leoposc",password="4xhm!zMY@dAVNQ7"
# database="weatherData"


def get_timezone(location: str):
    # initialize geolocator
    geolocation = Nominatim(user_agent="leopldSchmid")
    # get timezone for location
    location = geolocation.geocode(location)
    tzfinder = TimezoneFinder()
    timezone = tzfinder.timezone_at(lng=location.longitude, lat=location.latitude)
    return timezone


def validate_time_range(start: str, end: str, start_timestamp_database: int, end_timestamp_database: int):
    # convert start to timestamp, timezone is local time
    start_timestamp_requested = int(datetime.strptime(start, '%Y-%m-%d').replace(hour=0, minute=0, second=0).timestamp())
    # convert end to timestamp from this day at 23:00:00
    end_timestamp_requested = int(datetime.strptime(end, '%Y-%m-%d').replace(hour=23, minute=0, second=0).timestamp())

    # check if the requested time range is already in the database
    if start_timestamp_requested >= start_timestamp_database and end_timestamp_requested <= end_timestamp_database:
        raise Exception("The requested time range is already in the database.")
    
    # check if the ending of the requested time range is in the database, but not the beginning
    elif start_timestamp_requested < start_timestamp_database and end_timestamp_requested > start_timestamp_database:
        # set end_timestamp_requested to one day before the start_timestamp_database
        end_timestamp_requested = start_timestamp_database - 86400
        # set end to one day before the start (UTC)
        end = datetime.fromtimestamp(end_timestamp_requested).strftime('%Y-%m-%d')
        print("The ending of the requested time range is already in the database.")

    #check if beginning of requested time range is in the database, but not the end
    elif start_timestamp_requested < end_timestamp_database and end_timestamp_requested > end_timestamp_database:
        # set start_timestamp_requested to one day after the end_timestamp_database
        start_timestamp_requested = end_timestamp_database + 86400
        # set start to one day after the end
        start = datetime.fromtimestamp(start_timestamp_requested).strftime('%Y-%m-%d')
        print("The beginning of the requested time range is already in the database.")
    return start, end


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

        location = location.replace(" ", "__").strip()

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
    def create_hp_table(cur, self):

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS hyperparameters_machinelearning (
            solarsystem_id SERIAL PRIMARY KEY,
            power_index SMALLINT,
            model_type VARCHAR(50),
            max_depth SMALLINT,
            max_features VARCHAR(20),
            min_samples_leaf SMALLINT, 
            max_leaf_nodes INTEGER,
            min_weight_fraction_leaf FLOAT,
            gamma FLOAT,
            learning_rate FLOAT
        );""")


    @connect
    def fetch_solar_data(cur, self, solarsystem_id: int, start: str, end: str):

        # table_name = ("solardata_"+location.strip()+"_"+str(solarsystem_id)).lower()

        power_index = self.get_power_index(solarsystem_id)

        # find existing tables in database
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        tables = cur.fetchall()

        table_name = 'Unknown'
        for table in tables:
            if table[0].split('_')[-1] == str(solarsystem_id):
                table_name = table[0]
                break                

        # check if table already exists
        if table_name == 'Unknown':
            solar_data, location, power_index = scrape_solar_data(start, end, solarsystem_id, power_index)
            # create table
            self.create_table(location, "SOLAR", solarsystem_id)

        else:
            # sort the table by timeEpoch
            cur.execute("""
                SELECT timeepoch FROM %s ORDER BY timeepoch ASC
            """, (AsIs(table_name),))
            sorted_table = cur.fetchall()

            if len(sorted_table) != 0:
                start_timestamp_database = sorted_table[0][0]
                end_timestamp_database   = sorted_table[-1][0]
                start, end = validate_time_range(start, end, start_timestamp_database, end_timestamp_database)

            solar_data, location, power_index = scrape_solar_data(start, end, solarsystem_id, power_index)

        if power_index is not None:
            self.save_power_index(solarsystem_id, power_index)

        for row in solar_data:

            if not row or row[0] == 0:
                continue

            try:
                # insert row into db
                cur.execute("""
                    INSERT INTO %s (timeepoch, energyoutput)
                    VALUES (%s, %s)
                """, (AsIs(table_name), row[0], row[1]))
            except psycopg2.errors.UniqueViolation:
                print(f"Data for timestamp {row[0]} already exists in the database.")
                # handle psycopg2.errors.UniqueViolation. Most likely because of time change 
                # in march and october.
                conn = cur.connection
                conn.rollback()
                continue
            

    @connect
    def fetch_weather_data(cur, self,location: str,start: str, end: str):

        loc = location.replace(" ", "__").strip()

        table_name = ("weatherdata_"+loc).lower()

        # find existing tables in database
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        tables = cur.fetchall()

        # check if table already exists
        if (table_name,) not in tables:
            # create table
            self.create_table(location=location,table_type="WEATHER")

        # sort the table by timeEpoch
        cur.execute("""
            SELECT timeepoch FROM %s ORDER BY timeepoch ASC
        """, (AsIs(table_name),))
        sorted_table = cur.fetchall()

        if len(sorted_table) != 0:
            start_timestamp_database = sorted_table[0][0]
            end_timestamp_database   = sorted_table[-1][0]

            start, end = validate_time_range(start, end, start_timestamp_database, end_timestamp_database)
            
        key = "4MZDYZUR9MG5MTY4K8WJVT8K6"

        url = f'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{start}/{end}?unitGroup=metric&elements=datetime%2CdatetimeEpoch%2Ctemp%2Chumidity%2Cwindspeed%2Ccloudcover%2Csolarradiation%2Cuvindex&include=hours%2Cobs%2Cremote&key={key}&contentType=csv'
                
        result = requests.get(url)

        csv_data = result.text.split('\n')
        csv_reader = csv.reader(csv_data, delimiter=',')
        next(csv_reader)

        
                                                                                        #  csv_header = ['datetime',
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
            insert_row[0] = int(datetime.strptime(row[0], '%Y-%m-%dT%H:%M:%S').timestamp())

            # get hour from timestamp
            insert_row[1] = datetime.fromtimestamp(insert_row[0]).hour
            # get calendar week from timestamp
            insert_row[2] = datetime.fromtimestamp(insert_row[0]).isocalendar()[1]
            for i in range(1, len(row)-1):
                if row[i] == '':
                    # change missing values to None or 0 if it is the solar radiation
                    insert_row[i+2] = None if i != len(row)-2 else 0
                else:
                    # if i == len(row)-2:
                    #     # scale solar energy to 100
                    #     insert_row[i+2] = int(float(row[i])*100)
                    # else:
                    insert_row[i+2] = int(float(row[i]))          
            try:
                # insert row into db    
                cur.execute("""
                    INSERT INTO %s (timeepoch, 
                    hour, calendarweek, temperature, humidity, wind,
                    cloudcoverage, solarradiation)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (AsIs(table_name), *insert_row))
            except psycopg2.errors.UniqueViolation:
                print(f"Data for timestamp {insert_row[0]} already exists in the database.")
                # handle psycopg2.errors.UniqueViolation. Most likely because of time change 
                # in march and october.
                conn = cur.connection
                conn.rollback()
                continue


    @connect
    def select_solar_data(cur, self, solarsystem_id: str):
        
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        tables = cur.fetchall()

        table_name = 'Unknown'
        for table in tables:            
            if table[0].split('_')[-1] == str(solarsystem_id):
                table_name = table[0]
                break

        # table_name = ("solardata_"+location.strip()+"_"+str(solarsystem_id)).lower()

        # # check if table even exists
        # cur.execute("""
        #     SELECT table_name FROM information_schema.tables
        #     WHERE table_schema = 'public'
        # """)
        # tables = cur.fetchall()

        # assert (table_name,) in tables, f"The table '{table_name}' does not exist."
        assert table_name != 'Unknown', f"The table '{table_name}' does not exist."

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

        

        table_name = ('weatherdata_'+location.replace(' ','__').strip()).lower()

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
        df = pd.DataFrame(weather_data, columns=['timeepoch', 'hour', 'calendarweek', 'solarradiation', 'temperature', 'cloudcoverage', 'humidity', 'wind',])
        return df


    @connect
    def save_hypterparameters(cur, self, solarsystem_id: int, hyperparameters: dict, model_type: str):

        self.create_hp_table()

        model_type = model_type.split('.')[-1].split("'")[0].lower()

        # unpack keys and values of dict
        keys = list(hyperparameters.keys())
        values = list(hyperparameters.values())
        # convert None to -123
        values = [-123 if value is None else value for value in values]
        keys.append('solarsystem_id')
        values.append(solarsystem_id)
        keys.append('model_type')
        values.append(model_type)
    
        # check if hyperparameters for solarsystem already exist
        cur.execute(f"""
            SELECT solarsystem_id FROM hyperparameters_machinelearning
            WHERE solarsystem_id = {solarsystem_id}
        """)
        result = cur.fetchall()

        # if not insert hyperparameters into solarsystem
        if not result:

            clause = ', '.join(['%s']*len(keys))
            
            cur.execute(f"""
                INSERT INTO hyperparameters_machinelearning ({', '.join(keys)})
                VALUES ({clause})
            """, values)            
        
        # else update hyperparameters
        else:

            # set_clause = ', '.join([f'{key} = %s' for key in keys])

            # get power index from solarsystem
            cur.execute(f"""
                SELECT power_index FROM hyperparameters_machinelearning
                WHERE solarsystem_id = {solarsystem_id}
            """)
            power_index = cur.fetchall()[0][0]
            keys.append('power_index')
            values.append(power_index)

            # delete row 
            cur.execute(f"""
                DELETE FROM hyperparameters_machinelearning
                WHERE solarsystem_id = {solarsystem_id}
            """)           

            cur.execute(f"""
                INSERT INTO hyperparameters_machinelearning ({', '.join(keys)})
                VALUES ({', '.join(['%s']*len(keys))})
            """, values)


    @connect
    def save_power_index(cur, self, solarsystem_id: int, power_index: int):

        self.create_hp_table()

        # check if row already exists
        cur.execute(f"""
            SELECT solarsystem_id FROM hyperparameters_machinelearning
            WHERE solarsystem_id = {solarsystem_id}
        """)
        result = cur.fetchall()

        # if not insert row
        if not result:
            cur.execute(f"""
                INSERT INTO hyperparameters_machinelearning (solarsystem_id, power_index)
                VALUES ({solarsystem_id}, {power_index})
            """)

        # else update row
        else:
            
            # insert power index into solarsystem
            cur.execute(f"""
                UPDATE hyperparameters_machinelearning
                SET power_index = {power_index}
                WHERE solarsystem_id = {solarsystem_id}
            """)
        


    @connect
    def get_power_index(cur, self, solarsystem_id: int):
               
        # get power index from solarsystem
        cur.execute(f"""
            SELECT power_index FROM hyperparameters_machinelearning
            WHERE solarsystem_id = {solarsystem_id}
        """)
        
        try:
            power_index = cur.fetchone()[0]
        except TypeError:
            power_index = None

        return power_index
    

    @connect
    def get_hyperparameters(cur, self, solarsystem_id: int):
            
            # check if table exists
            # cur.execute("""
            #     SELECT hyperparameters_machinelearning FROM information_schema.tables
            #     WHERE table_schema = 'public'
            # """)
            # table = cur.fetchall()


            # get hyperparameters from db
            cur.execute(f"""
                SELECT * FROM hyperparameters_machinelearning
                WHERE solarsystem_id = {solarsystem_id}
            """)
            hyperparameters = cur.fetchone()
            # fetch column names from db, sorted by ordinal position
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'hyperparameters_machinelearning'
                ORDER BY ordinal_position
            """)

            keys = cur.fetchall()
            keys = [key[0] for key in keys]

            if hyperparameters is None:
                return None
            
            # create dict from column names and hyperparameters
            hyperparameters = dict(zip(keys, hyperparameters))
            # drop solarsystem_id and power_index
            hyperparameters.pop('solarsystem_id')
            hyperparameters.pop('power_index')
            # pop None values
            hyperparameters = {key: value for key, value in \
                hyperparameters.items() if value is not None}
            # convert values with -123 to None
            hyperparameters = {key: None if value == -123 or value == '-123' else \
                value for key, value in hyperparameters.items()}

            return hyperparameters
    

    



        


# manager = DBManager()
# manager.fetch_weather_data('Cherry Hill Township', '2019-01-01', '2019-01-01')
# manager.create_hp_table()
# manager.fetch_solar_data(10, "2022-05-30", "2022-06-13")
# manager.save_hypterparameters(1200, p)
# print(manager.get_hyperparameters(1200))