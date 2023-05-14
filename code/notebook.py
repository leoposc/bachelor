#%%
from db import DBManager


# fetch data from servers
manager = DBManager()
manager.fetch_solar_data("2022-01-05", "2022-01-06", 10)
# manager.fetch_weather_data("2022-01-01", "2022-01-31", "applewood")


# %%
