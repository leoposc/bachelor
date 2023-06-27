# %%
from db import DBManager
import matplotlib.pyplot as plt
import pandas as pd
import psycopg2
from datetime import datetime, timedelta


class Tester():

    def __init__(self):
        self.dm = DBManager()

    def test_solar_data_consistency(self, solarsystem_id):
        solar_data = self.dm.select_solar_data(solarsystem_id)
        print(f"Checking {len(solar_data)} entries for consistency")
        # check for consistency
        sum_errors = 0
        for i in range(1, len(solar_data)-1):
            # check if the next entry is 1 hour later
            # print(solar_data[i])
            if solar_data['timeepoch'][i] + 3600 != solar_data['timeepoch'][i+1]:
                print("\nInconsistent data at index: ", i)
                print(f"Timestamp: {solar_data['timeepoch'][i]} vs. {solar_data['timeepoch'][i+1]}")
                time1 = pd.to_datetime(solar_data['timeepoch'][i], unit='s')
                time2 = pd.to_datetime(solar_data['timeepoch'][i+1], unit='s')
                print(f"Datetime: {time1} vs. {time2}")
                sum_errors += 1

        print(f"\nFound {sum_errors} inconsistencies")


    def test_weather_data_consistency(self, location: str):
        weather_data = self.dm.select_weather_data(location)
        print(f"Checking {len(weather_data)} entries for consistency")
        # check for consistency
        sum_errors = 0
        for i in range(1, len(weather_data)-1):
            # check if the next entry is 1 hour later
            if weather_data['timeepoch'][i] + 3600 != weather_data['timeepoch'][i+1]:
                print("\nInconsistent data at index: ", i)
                print(f"Timestamp: {weather_data['timeepoch'][i]} vs. {weather_data['timeepoch'][i+1]}")
                time1 = pd.to_datetime(weather_data['timeepoch'][i], unit='s')
                time2 = pd.to_datetime(weather_data['timeepoch'][i+1], unit='s')
                print(f"Datetime: {time1} vs. {time2}")
                sum_errors += 1

        print(f"\nFound {sum_errors} inconsistencies")



    def verify_db_timestamps(self, table_name: str):
        # Connect to the database
        conn = psycopg2.connect(database='bachelor')
        cursor = conn.cursor()

        # table_name = 'solar_data_copy'

        # Execute a query to fetch all rows from the table
        cursor.execute(f'SELECT * FROM {table_name}')

        invalid_rows = 0
        valid_rows = 0
        updated_rows = 0

        # Iterate over each row
        for row in cursor.fetchall():
            # convert the timestamp to a datetime object
            timestamp = datetime.fromtimestamp(row[0])
            energyoutput = row[1]

            # Check if the timestamp value represents a straight hour (0-23)
            if timestamp.minute == 0 and timestamp.second == 0:
                print(f"Row {timestamp} is valid.")
                valid_rows += 1
            else:
                # Round the timestamp to the next closest hour
                if timestamp.minute >= 30:
                    rounded_timestamp = timestamp.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                else:
                    rounded_timestamp = timestamp.replace(minute=0, second=0, microsecond=0)
                # rounded_timestamp = timestamp.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)

                # delete the row
                cursor.execute(f'DELETE FROM {table_name} WHERE timeepoch = %s', (int(timestamp.timestamp()),))

                # conn.commit()

                # Insert a new row with the rounded timestamp
                try:
                    cursor.execute(f'INSERT INTO {table_name} (timeepoch, energyoutput) VALUES (%s, %s)', (int(rounded_timestamp.timestamp()), energyoutput))
                except psycopg2.errors.UniqueViolation:
                    print(f"Row {timestamp} has already been updated with timestamp {rounded_timestamp}. Update it instead")
                    cursor.execute(f'UPDATE {table_name} SET energyoutput = %s WHERE timeepoch = %s', (energyoutput, int(timestamp.timestamp())))
                    updated_rows += 1

                print(f"Row {timestamp} has been updated with timestamp {rounded_timestamp}.")
                invalid_rows += 1

        print(f"Found {invalid_rows} invalid rows and {valid_rows} valid rows. Updated {updated_rows} rows.")
        conn.commit()



# %%
tester = Tester()
# tester.test_solar_data_consistency(1200)
tester.test_weather_data_consistency('linthicum')
# tester.verify_db_timestamps('solardata_linthicum_1200')



# %%
