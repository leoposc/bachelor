#%%
from db import DBManager

# fetch data from servers
manager = DBManager()

manager.fetch_solar_data(10, "2022-05-30", "2022-07-10")
# manager.fetch_weather_data("applewood", "2022-05-30", "2022-07-10")

# manager.fetch_solar_data(1200, "2019-01-01", "2019-03-01")
# manager.fetch_weather_data("linthicum", "2019-02-01", "2019-02-15")


# %%

from scikit import ScikitManager

sci = ScikitManager(location='applewood', solarsystem_id=10)
# sci = ScikitManager(location='linthicum', solarsystem_id=1200)
sci.get_data()
sci.calculate_energyoutput_index()
sci.choose_features(['solarradiation',
                     'energyoutput_index',
                    #  'temperature',
                    #  'wind',
                    #  'humidity',
                     'energyoutput'
                     ])
# sci.choose_features(['solarradiation', 'temperature', 'energyoutput'])
# sci.choose_features(['solarradiation', 'energyoutput'])
# sci.compare_similar_radiation(470, 550)
# sci.filter_low_radiation()
# sci.split_data(0.1)
sci.update_numpy_arrays()
#%%
# sci.choose_features(['solarradiation', 'temperature', 'wind', 'energyoutput'])
# %%
# sci.standardise()11
# sci.normalise()

sci.visualize_pairwise_correlation()

sci.visualize_heatmap()


# %%

# sci.fit('LinearRegression')
# sci.fit('Ridge')
# sci.fit('Lasso')
# sci.fit('ElasticNet')
# sci.fit('SVR')
sci.fit('DecisionTreeRegressor')
sci.predict()
sci.evaluate()

# %%

sci.visualize_residues()

#%%
import pandas as pd 

# sci.get_data()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(sci.XY_df)



# %%
print(sci.X_train.shape)
print(sci.y_train.shape)
print(sci.X_test.shape)
print(sci.y_test.shape)
print(sci.XY_df)
# %%

for i in range(len(sci.y_test)):
    print('Actual value: {:>10.2f}, Predicted value: {:>10.2f}'.format(float(sci.y_test[i]), float(sci.y_test_pred[i])))

# %%
sci.predict()
sci.evaluate()
# %%

