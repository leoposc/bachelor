from db import DBManager 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import matplotlib.pyplot as plt
from itertools import product
import seaborn as sns
import pandas as pd
import numpy as np


def timer(start_time=None):
    if not start_time:
        start_time=datetime.now()
        return start_time
    elif start_time:
        thour,temp_sec=divmod((datetime.now()-start_time).total_seconds(),3600)
        tmin,tsec=divmod(temp_sec,60)
        print("Computation time of optimal hyperparameter grid search: ", thour,":",tmin,':',round(tsec,2))


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


# get index of local maxima within a range of 7 entries
def get_local_maxima_index(np_series: np.array):
    local_maxima = []
    for i in range(3, len(np_series)-3):
        if np_series[i] > np_series[i-3] and np_series[i] > np_series[i-2] \
            and np_series[i] > np_series[i-1] and np_series[i] > np_series[i+1] \
            and np_series[i] > np_series[i+2] and np_series[i] > np_series[i+3]:
            local_maxima.append(i)
    return local_maxima
    

class ScikitManager():
    XY_df : pd.DataFrame
    X_train : np.ndarray
    y_train : np.ndarray
    X_test : np.ndarray
    y_test : np.ndarray
    y_train_pred : np.ndarray
    y_test_pred : np.ndarray
    timeepoch_train: np.ndarray
    timeepoch_test: np.ndarray

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
        column_order.remove('energyoutput')      # Remove 'energyoutput' from the column order
        column_order.append('energyoutput')      # Append 'energyoutput' at the end
        column_order.remove('timeepoch')         # Remove 'timeepoch' from the column order
        column_order.insert(0, 'timeepoch')      # Insert 'timeepoch' at the beginning
        self.XY_df = self.XY_df.reindex(columns=column_order)   # Reindex the DataFrame with the new column order
        self.X_train = self.XY_df.iloc[:, 1:-1].values
        self.y_train = self.XY_df.iloc[:,  -1].values.reshape(-1,1)
        self.timeepoch_train = self.XY_df.iloc[:, 0].values.reshape(-1,1)


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
        self.XY_df['timeepoch'] = pd.to_datetime(self.XY_df['timeepoch'], unit='s')      
        # # drop timeepoch column
        # self.XY_df   = self.XY_df.drop(columns=['timeepoch'])
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
        self.X_train, self.X_test, self.y_train, self.y_test, self.timeepoch_train,\
            self.timeepoch_test = train_test_split(self.X_train, self.y_train, \
            self.timeepoch_train, test_size=test_size, random_state=0)
        


    def split_data_by_days(self, test_size):
        # Find unique dates from 'timeepoch' column
        # unique_dates = pd.to_datetime(self.XY_df['timeepoch'], unit='s').dt.date.unique()
        unique_dates = self.XY_df['timeepoch'].dt.date.unique()
        
        # Shuffle the unique dates
        shuffled_dates = unique_dates.copy()
        np.random.shuffle(shuffled_dates)
        
        # Split the shuffled dates into train and test sets
        train_dates, test_dates = train_test_split(shuffled_dates, test_size=test_size, shuffle=False)
        
        # Filter the DataFrame based on train and test dates
        train_df = self.XY_df[self.XY_df['timeepoch'].dt.date.isin(train_dates)]
        test_df = self.XY_df[self.XY_df['timeepoch'].dt.date.isin(test_dates)]

        self.X_train = train_df.drop(columns=['timeepoch','energyoutput']).to_numpy()
        self.y_train = train_df['energyoutput'].to_numpy().flatten()
        self.X_test = test_df.drop(columns=['timeepoch','energyoutput']).to_numpy()
        self.y_test = test_df['energyoutput'].to_numpy().flatten()
        self.timeepoch_train = train_df['timeepoch'].to_numpy().flatten()           
        self.timeepoch_test = test_df['timeepoch'].to_numpy().flatten()      
        
    

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


    def grid_search(self):
        dm = DBManager()
        parameters={
                    # "splitter":["best","random"],
                    "max_depth" : [8,9,11,12,16],
                    "min_samples_leaf":[3,4,5,6,7],
                    "min_weight_fraction_leaf":[0.0, 0.1,0.2,0.3],
                    "max_features":["auto","log2","sqrt",None],
                    "max_leaf_nodes":[None,8,16,32,64,128,256,512]
                    }
        model_type = str(type(self.model))

        start_time = timer(None)
        tuning_model = GridSearchCV(self.model, param_grid=parameters,scoring='neg_mean_squared_error',cv=3,verbose=2)
        tuning_model.fit(self.X_train,self.y_train)
        timer(start_time=start_time)

        self.model = tuning_model
        dm.save_hypterparameters(self.solarsystem_id, self.model.best_params_, model_type=model_type)
        text = [f'{key}: {value} \n' for key, value in self.model.best_params_.items()]
        print(f"Grid search finished. Following are the best hyperparameters:\n\n{''.join(text)}")


    # def fit(self, model_type=None):
    #     dm = DBManager()

    #     self.hyperparams = dm.get_hyperparameters(self.solarsystem_id)

        
        
    #     self.model_selection()
        
        
        


    def make_sets(self):
        c = 0.001
        gamma = 1e-10
        param_grid = {
            "C": [c*(10**i) for i in range(1, 10)],
            "gamma": [gamma*(10**i) for i in range(1, 10)],
            "max_depth" : [i for i in range(1, 10)]            
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




    def model_selection(self, model_type=None):
        dm = DBManager()
        self.hyperparams = dm.get_hyperparameters(self.solarsystem_id)

        if self.hyperparams is not None and (model_type is None or \
            model_type.lower() == self.hyperparams['model_type']):
            self.hyperparams.pop('model_type')            
            text = [f'{key}: {value} \n' for key, value in self.hyperparams.items()]
            print(f"Found hyperparameters for this model. Using:\n\n{''.join(text)}")

        else:
            self.hyperparams = None
            print('No hyperparameters found for this model. Using default hyperparameters.')


        if model_type == 'LinearRegression':
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression(**self.hyperparams) if self.hyperparams else LinearRegression()
        elif model_type == 'Polynomial':
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression
            self.model = LinearRegression(**self.hyperparams) if self.hyperparams else LinearRegression()
            poly = PolynomialFeatures(degree=2)
            self.X_train = poly.fit_transform(self.X_train)
            self.X_test = poly.transform(self.X_test)
            
        elif model_type == 'Ridge':
            from sklearn.linear_model import Ridge
            self.model = Ridge(**self.hyperparams) if self.hyperparams else Ridge()
        elif model_type == 'Lasso':
            from sklearn.linear_model import Lasso
            self.model = Lasso(**self.hyperparams) if self.hyperparams else Lasso()
        elif model_type == 'ElasticNet':
            from sklearn.linear_model import ElasticNet
            self.model = ElasticNet(**self.hyperparams) if self.hyperparams else ElasticNet()
        elif model_type == 'decisiontreeregressor':
            from sklearn.tree import DecisionTreeRegressor
            self.model = DecisionTreeRegressor(**self.hyperparams) if self.hyperparams else DecisionTreeRegressor()
        elif model_type == 'RandomForestRegressor':
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(**self.hyperparams) if self.hyperparams else RandomForestRegressor()
        elif model_type == 'SVR':
            from sklearn.svm import SVR
            self.model = SVR(**self.hyperparams) if self.hyperparams else SVR()
        elif model_type == 'KNeighborsRegressor':
            from sklearn.neighbors import KNeighborsRegressor
            self.model = KNeighborsRegressor(**self.hyperparams) if self.hyperparams else KNeighborsRegressor()
        # elif model_type == 'XGBRegressor':
        #     from xgboost import XGBRegressor
        #     self.model = XGBRegressor()
        else:
            raise Exception('No model type specified in database. Please specify one.')
            
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
        max_len = min(len(self.y_train_pred), len(self.y_train), len(self.y_test_pred), len(self.y_test))
        y_train_residuals = self.y_train_pred - self.y_train.flatten()
        y_test_residuals = self.y_test_pred - self.y_test.flatten()        

        plt.scatter(self.y_train_pred, y_train_residuals, c='steelblue', 
                    edgecolor='white', marker='o', s=35, alpha=0.9, label='training data')
        plt.scatter(self.y_test_pred, y_test_residuals, c='limegreen',
                    marker='s', s=35, alpha=0.9, label='test data')
        plt.xlabel('Predicted values')
        plt.ylabel('Residuals')
        plt.legend(loc='upper left')
        plt.hlines(y=0, lw=2, color='black', xmin=min(self.y_train_pred), xmax=max(self.y_train_pred))
        plt.rcParams['figure.figsize'] = [20, 10]
        plt.show()


    def visualize_predictions(self, number_entries=None):
        if number_entries is not None and number_entries < len(self.y_test):
            y_test = self.y_test[:number_entries]
            y_test_pred = self.y_test_pred[:number_entries]
        else:
            y_test = self.y_test
            y_test_pred = self.y_test_pred            
        # print(self.y_test.shape)
        # print(self.y_test_pred.shape)
        date_indices = get_local_maxima_index(self.y_test_pred)
        # convert timeepoch to date
        dates = [str(x)[:10] for x in self.timeepoch_test[date_indices]]
        # plt.xticks(range(len(self.y_test_pred)), dates, rotation=45)
        # plt.plot(range(len(y_test)), y_test, label='Actual')
        # plt.plot(range(len(y_test_pred)), y_test_pred, label='Predicted')
        plt.plot(self.timeepoch_test[:len(y_test)], y_test, label='Actual')
        plt.plot(self.timeepoch_test[:len(y_test)], y_test_pred, label='Predicted')
        plt.xticks(rotation=45)
        
        plt.title(f'Solarsystem id: {self.solarsystem_id}, Location: {self.location}')
        plt.xlabel('Time')
        plt.ylabel('Solar energy production')
        plt.legend(loc='upper left')
        # plt.rcParams['figure.figsize'] = [20, 10]
        plt.show()


    