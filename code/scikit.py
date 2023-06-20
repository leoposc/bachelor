from db import DBManager 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from itertools import product
import seaborn as sns
import pandas as pd
import numpy as np


def consider_wind(ds: pd.Series):  
    # filter temperature values above 20, since only then
    # wind as natural fan could have an effect on the energy output
    if ds['temperature'] > 20:    
        return ds['temperature'] - (ds['wind'] * 1.0)
    else:
        return ds['temperature']


def consider_temperature(df: pd.DataFrame):
    energyoutput_index = df['solarradiation'] - (df['temperature'] * 5.0)
    return energyoutput_index
    

class ScikitManager():
    XY_df : pd.DataFrame
    X_train : np.ndarray
    y_train : np.ndarray
    X_test : np.ndarray
    y_test : np.ndarray
    y_train_pred : np.ndarray
    y_test_pred : np.ndarray

    model: None
    features: list
    location: str
    solarsystem_id: int
    start: str
    end: str


    def __init__(self, location: str, solarsystem_id: int, 
                 start=None, end=None):
        self.location = location
        self.solarsystem_id = solarsystem_id
        self.start = start
        self.end = end


    def calculate_energyoutput_index(self):
        self.XY_df['temperature'] = self.XY_df.apply(consider_wind, axis=1)
        self.XY_df['energyoutput_index'] = self.XY_df.apply(consider_temperature, axis=1)

    # def fetch_data(self):
    #     dm = DataManager()
    #     dm.fetch_weather_data(self.location, self.start, self.end)
    #     dm.fetch_solar_data(self.location, self.solarsystem_id, self.start, self.end)


    def update_numpy_arrays(self):        
        column_order = list(self.XY_df.columns)  # Get the current column order
        column_order.remove('energyoutput')  # Remove 'energyoutput' from the column order
        column_order.append('energyoutput')  # Append 'energyoutput' at the end
        self.XY_df = self.XY_df.reindex(columns=column_order)  # Reindex the DataFrame with the new column order
        self.X_train = self.XY_df.iloc[:, :-1].values
        self.y_train = self.XY_df.iloc[:,  -1].values.reshape(-1,1)


    def update_panda_dataframe(self):        
        features = self.features.copy()
        # drop energyoutput column
        features.remove('energyoutput')
        self.XY_df = pd.DataFrame(self.X_train, columns=features)
        self.XY_df['energyoutput'] = self.y_train


    def get_data(self):
        dm = DBManager()
        solar_df = dm.select_solar_data(self.solarsystem_id)
        weather_df = dm.select_weather_data(self.location)     
        # merge data on timeepoch
        self.XY_df   = weather_df.merge(solar_df, on='timeepoch')
        # drop timeepoch column
        self.XY_df   = self.XY_df.drop(columns=['timeepoch'])
        # print size of dataset
        print(f"Solar dataset size: {solar_df.shape}")
        print(f"Weather dataset size: {weather_df.shape}")
        print(f"Dataset size: {self.XY_df.shape}")


    def filter_low_radiation(self):
        # filter out rows with low radiation
        self.XY_df   = self.XY_df[self.XY_df['solarradiation'] > 10]


    def filter_low_energyoutput(self):
        # filter out rows with low energy output
        self.XY_df   = self.XY_df[self.XY_df['energyoutput'] > 10]


    def compare_similar_radiation(self, lower_limit: int, upper_limit: int):
        # filter out rows with similar radiation
        self.XY_df   = self.XY_df[self.XY_df['solarradiation'] > lower_limit]
        self.XY_df   = self.XY_df[self.XY_df['solarradiation'] < upper_limit]


    def choose_features(self, features: list):
        self.features = features
        self.XY_df = self.XY_df[features]


    def standardise(self):
        stsc = StandardScaler()
        self.X_train = stsc.fit_transform(self.X_train)
        self.X_test = stsc.transform(self.X_test)
        

    def normalise(self):
        mmsc = MinMaxScaler()
        self.X_train = mmsc.fit_transform(self.X_train)
        self.X_test = mmsc.transform(self.X_test)


    def split_data(self, test_size: float):
        # warning: check if this is the correct way to split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_train, self.y_train, test_size=test_size, random_state=0)



    # def transform_features(self):
    #     # perform log transformation on solarradiation
    #     self.X_df['solarradiation'] = np.log(self.X_df['solarradiation'])
    #     # get square root of energyoutput
    #     self.y_df['energyoutput'] = np.sqrt(self.y_df['energyoutput'])
    #     # self.y_df['energyoutput'] = self.y_df['energyoutput'].apply(np.sqrt)
    #     # update numpy arrays
    #     self.X_train = self.X_df.values
    #     self.y_train = self.y_df.values.reshape(-1,1)


    def visualize_pairwise_correlation(self):
        sns.pairplot(self.XY_df[self.features], height=2.5)
        plt.tight_layout()
        plt.show()


    def visualize_heatmap(self):
        cm = np.corrcoef(self.XY_df[self.features].values.T)
        sns.set(font_scale=1.5)
        hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=self.features, xticklabels=self.features)
        plt.show()


    def visualize_data_range(self):
        data = self.XY_df[self.features]
        # drop energyoutput
        data = data.drop(columns=['energyoutput'])
        g = sns.boxplot(data=data, linewidth=2.5)
        # g.set_yscale("log")


    def make_sets(self):
        c = 0.001
        gamma = 1e-10
        param_grid = {
            "C": [c*(10**i) for i in range(1, 14)],
            "gamma": [gamma*(10**i) for i in range(1, 14)]
        }

        sets = list()
        all_hps_vals = [lst for lst in param_grid.values()]
        hp_keys = [hp for hp in param_grid.keys()]
        val_sets = product(*all_hps_vals)
        for val in val_sets:
            hp_set = dict()
            for idx, hp_key in enumerate(hp_keys):
                hp_set[hp_key] = val[idx]
            sets.append(hp_set)

        self.hp_sets = sets


    def grid_search(self,model_type='LinearRegression'):
        self.make_sets()
        best_score = 0
        best_params = None
        for hp_set in self.hp_sets:
            
            self.fit(model_type)            
            if self.score > best_score:
                best_score = self.score
                best_params = hp_set
        print(f"Best score: {best_score}")
        print(f"Best params: {best_params}")


    def grid_search_v2(self):
        self.make_sets()
        logs = list()
        best_hp_set = {
            "best_test_score": 0.0
        }
        for hp_set in self.hp_sets:
            log = dict()
            self.model = self.model(**hp_set)
            self.model.fit(self.X_train, self.y_train.flatten())
            train_score = self.model.score(self.X_train, self.y_train)
            test_score = self.model.score(self.X_test, self.y_test)

            log["hp"] = hp_set
            log["train_score"] = train_score
            log["test_score"] = test_score

            if test_score > best_hp_set["best_test_score"]:
                best_hp_set["best_test_score"] = test_score
                best_hp_set["best_hp"] = hp_set

            logs.append(log)




    def fit(self, model_type='LinearRegression'):
        if model_type == 'LinearRegression':
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression()
        elif model_type == 'Polynomial':
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression()
            poly = PolynomialFeatures(degree=2)
            self.X_train = poly.fit_transform(self.X_train)
            self.X_test = poly.transform(self.X_test)
            
        elif model_type == 'Ridge':
            from sklearn.linear_model import Ridge
            self.model = Ridge()
        elif model_type == 'Lasso':
            from sklearn.linear_model import Lasso
            self.model = Lasso()
        elif model_type == 'ElasticNet':
            from sklearn.linear_model import ElasticNet
            self.model = ElasticNet()
        elif model_type == 'DecisionTreeRegressor':
            from sklearn.tree import DecisionTreeRegressor
            self.model = DecisionTreeRegressor()
        elif model_type == 'RandomForestRegressor':
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor()
        elif model_type == 'SVR':
            from sklearn.svm import SVR
            self.model = SVR()
        elif model_type == 'KNeighborsRegressor':
            from sklearn.neighbors import KNeighborsRegressor
            self.model = KNeighborsRegressor()
        # elif model_type == 'XGBRegressor':
        #     from xgboost import XGBRegressor
        #     self.model = XGBRegressor()
        else:
            print('Invalid model type')
            return

        self.model.fit(self.X_train, self.y_train.flatten())
        # self.score = self.model.score(self.X_test, self.y_test)


    def analyze_feature_importance(self):
        feat_importances = self.model.feature_importances_
        indices = self.features.remove('energyoutput')
        feat_importances = pd.Series(feat_importances, index=indices)
        plt.show()

    
    def predict(self):
        self.y_test_pred = self.model.predict(self.X_test)
        self.y_train_pred = self.model.predict(self.X_train)


    def evaluate(self):
        print('Mean squared error - Test: %.2f, Training: %.2f'
               % (mean_squared_error(self.y_test, self.y_test_pred),
                  mean_squared_error(self.y_train, self.y_train_pred)))
        print('Coefficient of determination - Test: %.2f, Training: %.2f'
               % (r2_score(self.y_test, self.y_test_pred),
                  r2_score(self.y_train, self.y_train_pred)))


    def visualize_residues(self):
        plt.scatter(self.y_train_pred, self.y_train_pred - self.y_train, c='steelblue', 
                    edgecolor='white', marker='o', s=35, alpha=0.9, label='training data')
        plt.scatter(self.y_test_pred, self.y_test_pred - self.y_test, c='limegreen',
                    marker='s', s=35, alpha=0.9, label='test data')
        plt.xlabel('Predicted values')
        plt.ylabel('Residuals')
        plt.legend(loc='upper left')
        plt.hlines(y=0, lw=2, color='black')
        plt.show()