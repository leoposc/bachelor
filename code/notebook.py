#%%
from db import DBManager

# fetch data from servers
manager = DBManager()

# =============================================================================
# APPLEWOOD, 10
# =============================================================================
manager.fetch_solar_data(10, "2022-11-23", "2022-12-20")
manager.fetch_weather_data("applewood", "2022-11-11", "2022-12-20")

# =============================================================================
# LINTHICUM, 1200
# =============================================================================
# manager.fetch_solar_data(1200, "2019-04-13", "2019-06-30")
# manager.fetch_weather_data("linthicum", "2019-04-13", "2019-06-30")


# =============================================================================
# COCKEYSVILLE, 1201
# =============================================================================
# manager.fetch_solar_data(1201, "2019-02-06", "2019-03-15")
# manager.fetch_weather_data("Cherry Hill Townhill", "2019-02-06", "2019-03-15")
# manager.fetch_weather_data("Cherry Hill Townhill", "2019-02-06", "2019-03-15")

# =============================================================================
# NEW SMYRNA BEACH, 1231
# =============================================================================
# manager.fetch_solar_data(1231, "2004-05-01", "2004-06-30")
# manager.fetch_weather_data("New Smyrna beach", "2004-05-01", "2004-06-30")

# =============================================================================
# NEW SMYRNA BEACH, 1231
# =============================================================================
# manager.fetch_solar_data(1257, "2014-05-01", "2014-05-31")
# manager.fetch_weather_data("New Orleans", "2014-05-01", "2014-05-31")

# %%

# =============================================================================
# INITS
# =============================================================================

from scikit import ScikitManager
# sci = ScikitManager(location='applewood', solarsystem_id=10)
sci = ScikitManager(location='linthicum', solarsystem_id=1200)
# sci = ScikitManager(location='Cherry Hill Townhill', solarsystem_id=1201)
# sci = ScikitManager(location='New Smyrna beach', solarsystem_id=1231)


# %%

# sci = ScikitManager(location='linthicum', solarsystem_id=1200)
sci.get_data()

# sci.calculate_energyoutput_index()
sci.choose_features(['timeepoch',
                     'solarradiation',
                    #  'energyoutput_index',
                     'calendarweek',
                     'hour',
                     'temperature',
                    #  'wind',
                     'humidity',
                     'cloudcoverage',
                     'energyoutput'
                     ])
# sci.update_numpy_arrays()

#%%
sci.filter_low_energyoutput() 
# sci.split_data(0.1)
sci.split_data_by_days(0.2)
# sci.transform_hours()
# print(type(sci.timeepoch_test[0]))

#%%

sci.compare_similar_radiation(550, 600)

#%%

sci.model_selection('decisiontreeregressor')
# sci.grid_search()
sci.predict()
sci.evaluate()
# %%
# sci.visualize_residues()
# sci.visualize_predictions(xlim_left=400, xlim_right=700)
sci.visualize_predictions()
# sci.plot_outlier()

# sci.visualize_tree()
# sci.visualize_3d_plot()
#%%

sci.features
# sci.model.feature_importances_
# sci.model.n_features_in_
# sci.model.n_outputs_
# sci.model.tree_
# # dir(sci.model)

# %%

# sci.standardise()
# sci.normalise()
# sci.filter_low_radiation()
# %%

# sci.update_panda_dataframe()
# sci.update_numpy_arrays()
# sci.visualize_data_range()
sci.visualize_pairwise_correlation()
sci.visualize_heatmap()




#%%

sci.model_selection('decisiontreeregressor')
sci.grid_search()
# sci.fit()
# sci.predict()
# sci.evaluate()



# %%
print(sci.X_train.shape)
print(sci.y_train.shape)
print(sci.X_test.shape)
print(sci.y_test.shape)
# print(sci.XY_df)

#%%

# =============================================================================
# PLOT DATAFRAME
# =============================================================================

import pandas as pd 

# sci.get_data()
sci.update_panda_dataframe()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(sci.XY_df)
    # print(sci.X_test)

# %%

# PLOT PREDICTION VALUES
for i in range(len(sci.y_test)):
    print("Predicted value: %.1f, Real Value: %.1f" % (sci.y_test_pred[i], sci.y_test[i]))


for i in range(len(sci.y_test)):
    print('Actual value: {:>10.2f}, Predicted value: \
          {:>10.2f}'.format(float(sci.y_test[i]), float(sci.y_test_pred[i])))

# %%

import numpy as np
from datetime import datetime

# Check the type of the object before converting
print(sci.outliers_indices_test)
# [print(type(x.astype(datetime))) if isinstance(x, np.datetime64) else x for x in sci.timeepoch_test[[12,3,4,5,6,19]]]

# %%

sci.visualize_tree()
# %%
