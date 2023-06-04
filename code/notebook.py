#%%
from db import DBManager

# fetch data from servers
manager = DBManager()
manager.fetch_solar_data(10, "2022-05-01", "2022-05-31")
# manager.fetch_weather_data("applewood", "2022-05-01", "2022-05-31")


# %%

from scikit import ScikitManager

sci = ScikitManager(location='applewood', solarsystem_id=10)
sci.get_data()
sci.compare_similar_radiation(470, 550)
# sci.filter_low_radiation()
#%%

# sci.split_data(0.99)
# sci.standardise()
# sci.normalise()

sci.visualize_pairwise_correlation()

sci.visualize_heatmap()

#%%
import pandas as pd 

sci.get_data()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(sci.XY_df)



# %%
