from db import DBManager 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import export_graphviz
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from itertools import product
from datetime import datetime
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


# get index of local maxima within a range of 9 entries
# def get_local_maxima_index(np_series: np.array):
#     local_maxima = []
#     for i in range(4, len(np_series)-4):
#         if np_series[i] >= np_series[i-4] and np_series[i] >= np_series[i-3] and \
#             np_series[i] >= np_series[i-2] and np_series[i] >= np_series[i-1] and \
#             np_series[i] > np_series[i+1] and np_series[i] > np_series[i+2] and \
#             np_series[i] > np_series[i+3] and np_series[i] > np_series[i+4]:
#             local_maxima.append(i)
#     return local_maxima

# get index of local maxima within a range of x entries
def get_local_maxima_index(np_series: np.array, range_x=7):    
    tmp = ([0] * (range_x // 2) + list(np_series.copy()) + [0] * (range_x // 2))
    local_maxima = []
    for i in range((range_x) // 2, len(tmp)-((range_x+1) // 2)):
        for x in range(1, (range_x+1) // 2):
            if tmp[i] <= tmp[i-x] or tmp[i] <= tmp[i+x]:
                break
        else:   
            local_maxima.append(i)
    
    return np.array(local_maxima) - (range_x // 2)



# get index of local maxima within a range of x entries
# def get_local_maxima_index(np_series: np.array, range_x=7):    
#     tmp = ([0] * (range_x // 2) + list(np_series.copy()) + [0] * (range_x // 2))
#     local_maxima = []
#     for i in range((range_x) // 2, len(tmp)-((range_x+1) // 2), range_x):
#         for x in range(1, (range_x+1) // 2):
#             if tmp[i] <= tmp[i-x] or tmp[i] <= tmp[i+x]:
#                 break
#         else:   
#             local_maxima.append(i)
    
#     return np.array(local_maxima) - (range_x // 2)



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
        self.features = column_order
        self.XY_df = self.XY_df.reindex(columns=column_order)   # Reindex the DataFrame with the new column order
        self.X_train = self.XY_df.iloc[:, 1:-1].values
        self.y_train = self.XY_df.iloc[:,  -1].values.reshape(-1,1)
        self.timeepoch_train = self.XY_df.iloc[:, 0].values.reshape(-1,1)


    def update_panda_dataframe(self):        
        features = self.features.copy()
        # drop energyoutput and timeepoch column
        features.remove('energyoutput')
        features.remove('timeepoch')
        self.XY_df = pd.DataFrame(self.X_train, columns=features)
        self.XY_df['energyoutput'] = self.y_train
        self.XY_df['timeepoch'] = self.timeepoch_train


    def get_data(self):
        dm = DBManager()
        solar_df = dm.select_solar_data(self.solarsystem_id)
        weather_df = dm.select_weather_data(self.location)     
        # merge data on timeepoch
        self.XY_df   = weather_df.merge(solar_df, on='timeepoch')
        self.XY_df['timeepoch'] = pd.to_datetime(self.XY_df['timeepoch'], unit='s')              
        # print size of dataset
        print(f"Solar dataset size: {solar_df.shape}")
        print(f"Weather dataset size: {weather_df.shape}")
        print(f"Dataset size: {self.XY_df.shape}")
        # get maximum energy output
        print(f"Maximum energy output: {self.XY_df['energyoutput'].max()}")


    def filter_by(self, feature: str, lower_limit=None, upper_limit=None):
        # filter out rows which do not meet the criteria
        if lower_limit is not None:
            self.XY_df   = self.XY_df[self.XY_df[feature] > lower_limit]
        if upper_limit is not None:
            self.XY_df   = self.XY_df[self.XY_df[feature] < upper_limit]


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


    def analyze_cloudcover_quality(self):
        data = self.XY_df['cloudcoverage'].values
        min_unique  = 50 if len(data) < 350 else 70
        num_uniques = len(np.unique(data))
        # plot histogram
        print(f"Number of unique values: {num_uniques}")
        self.histogram_one_feature('cloudcoverage')
        if min_unique < num_uniques:
            print("Cloudcoverage kept in features")
        else:
            # remove cloudcoverage from features
            self.features.remove('cloudcoverage')
            self.XY_df = self.XY_df.drop(columns=['cloudcoverage'])
            print("Cloudcoverage removed from features")
        
            

    
    # warning: check if the column order stays in the same order
    # transform the feature hour into a polynomial feature 
    def transform_hours(self):
        poly = PolynomialFeatures(degree=4)
        # self.XY_df['hour'] = poly.fit_transform(self.XY_df['hour'].reshape(-1, 1))
        self.X_train = poly.fit_transform(self.X_train[:,2].reshape(-1, 1))
        self.X_test = poly.fit(self.X_test[:,2].reshape(-1, 1))


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


    def scatter_plot(self,features: list):
        if len(features) != 2:
            raise Exception('Please specify exactly two features to plot.')
        location = self.location[0].upper() + self.location[1:]
        fig = go.Figure(data=go.Scatter(x=self.XY_df[features[0]], y=self.XY_df[features[1]], mode='markers', marker=dict(size=3)))
        fig.update_layout(title=f'Location: {location}')
        features[0] = 'Daytime (h)'
        features[1] = 'Cloud coverage (%)'
        fig.update_xaxes(title_text=features[0])
        fig.update_yaxes(title_text=features[1])
        fig.show()

        # sns.kdeplot(data=self.XY_df, x=features[0], y=features[1], levels=100, cmap="Blues")
        # plt.show()


    def histogram_one_feature(self, feature: str):
        values = self.XY_df[feature].values
        plt.hist(values, bins=50, alpha=0.5)
        plt.xlabel('Cloud coverage (%)')
        plt.ylabel('Quantity')
        feature = feature[0].upper() + feature[1:]
        location = self.location[0].upper() + self.location[1:]
        plt.title(f'Histogram of {feature} in {location}', y=1.05)
        plt.rcParams.update({'font.size': 15})
        plt.tight_layout()
        plt.show()
        
        
    def visualize_pairwise_correlation(self):
        sns.set(font_scale=1.3)
        sns.pairplot(self.XY_df[self.features], height=2.5)
        plt.tight_layout()
        plt.show()


    def visualize_heatmap(self):
        features = self.features.copy()
        features.remove('timeepoch')
        plt.figure(figsize=(10,10))
        cm = np.corrcoef(self.XY_df[self.features].drop(columns=['timeepoch']).values.T)
        sns.set(font_scale=1.5)
        hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=features, xticklabels=features)
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
                    "max_depth" : [5,6,7,8,9,11,12,13,15,17],
                    "min_samples_leaf":[5,7,8,9,10,11,12,13,15,17,20,25,30,35,40,45,50],
                    # "min_weight_fraction_leaf":[0.0, 0.1,0.2,0.3],
                    "max_features":[None],
                    "max_leaf_nodes":[24,28,36,40,44,48,96,192,384,768,1536],
                    }
        model_type = str(type(self.model))

        start_time = timer(None)
        tuning_model = GridSearchCV(self.model, param_grid=parameters,scoring='neg_mean_squared_error',cv=3,verbose=1)
        tuning_model.fit(self.X_train,self.y_train)
        timer(start_time=start_time)

        self.model = tuning_model
        dm.save_hypterparameters(self.solarsystem_id, self.model.best_params_, model_type=model_type)
        text = [f'{key}: {value} \n' for key, value in self.model.best_params_.items()]
        print(f"Grid search finished. Following are the best hyperparameters:\n\n{''.join(text)}")


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

        if self.hyperparams is not None and 'model_type' in self.hyperparams and (model_type \
            is None or model_type.lower() == self.hyperparams['model_type']):
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
        # self.score = self.model.score(self.X_test, self.y_test


    def find_outlier(self):
        maxima_indices = np.array(get_local_maxima_index(self.y_test_pred))
        threshold = np.max(self.y_test[maxima_indices]) / 2
        outlier_indices = np.array(np.abs(self.y_test_pred[maxima_indices] \
            - self.y_test[maxima_indices]) > threshold)
        self.outliers_indices_test = maxima_indices[outlier_indices]
        # print(len(self.y_test))
        # print(len(self.outliers_indices_test))
        # print(len(outlier_indices))
        # print(outlier_indices)
        # print(self.outliers_indices_test)
        # outlier_indices = np.where(self.outliers_indices_test)[0]        
        # self.X_outlier_test = self.X_test[self.outliers_indices_test]
        # self.y_outlier_test = self.y_test[self.outliers_indices_test]        

        # maxima_indices = get_local_maxima_index(self.y_train)
        # threshhold_arr = self.y_train_pred[maxima_indices] - \
        #     (np.max(self.y_train[maxima_indices]) / 3)
        # outlier_indices = np.abs(self.y_train[maxima_indices] \
        #     - self.y_train_pred[maxima_indices]) > threshhold_arr
        # self.outliers_indices_train = maxima_indices[outlier_indices]
        # self.X_outlier_train = self.X_train[self.outliers_indices_train]
        # self.y_outlier_train = self.y_train[self.outliers_indices_train]


    # plot outliers and their surrouding data
    def plot_outlier(self, xlim_left=0, xlim_right=None):
        self.find_outlier()
        # abort if self.outliers_indices_test is empty
        if len(self.outliers_indices_test) == 0:
            print('\nNo outliers found.\n')
            return
        # enrichen outliers_indices with the surrounding 11 data points
        indices = np.array([[y for y in range(x-5,x+6)] for x in \
            self.outliers_indices_test]).flatten()

        rng = range(len(self.y_test[indices]))

        figure_outlier_idx = get_local_maxima_index(self.y_test_pred[indices], range_x=11)
        dates = [str(x)[:10] for x in self.timeepoch_test[self.outliers_indices_test]]
        
        plt.xticks(figure_outlier_idx, dates, rotation=65)
        plt.plot(rng, self.y_test[indices], label='solar energy prodcution')
        plt.plot(rng, self.y_test_pred[indices], label='solar energy prediction')        
        [plt.axvline(x=x_loc, color='red', linestyle='--')
            for x_loc in figure_outlier_idx]
        plt.plot([], [], color='red', linestyle='--', label='outlier')
        plt.title(f'Solarsystem id: {self.solarsystem_id}, Location: {self.location}')
        plt.ylabel('Solar energy production')
        plt.legend(loc='lower right')
        plt.xlim([xlim_left, xlim_right ])
        plt.rcParams['figure.figsize'] = [20, 10] 
        plt.show()
        


    def drum_off_outlier(self):
        threshold_arr = self.y_test_pred - (np.max(self.y_test) / 2)
        outliers_indices = self.y_test < threshold_arr
        new_dataset = self.X_test[~outliers_indices]
        new_labels = self.y_test[~outliers_indices]


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


    def visualize_residuals(self):
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
        plt.rcParams.update({'font.size': 32}) 
        plt.rcParams['figure.figsize'] = [20, 10]
        plt.show()


    def visualize_predictions(self, xlim_left=0, xlim_right=None):

        date_indices = get_local_maxima_index(self.y_test, range_x=11)        
        dates = [str(x)[:10] for x in self.timeepoch_test[date_indices]] # convert timeepoch to date
        fig_1 = plt.figure(figsize=(20,10))
        ax_1 = fig_1.add_axes([0.1,0.1,0.9,0.9])
        ax_1.set_xticks(date_indices)
        ax_1.set_xticklabels(dates, rotation=65)
        ax_1.plot(range(len(self.y_test)), self.y_test, label='Actual')
        ax_1.plot(range(len(self.y_test_pred)), self.y_test_pred, label='Predicted')        
        ax_1.legend(loc='upper left')
        plt.title(f'Solarsystem id: {self.solarsystem_id}, Location: {self.location}')
        plt.ylabel('Solar energy production')
        plt.rcParams.update({'font.size': 32}) 
        plt.rcParams['figure.figsize'] = [20, 10]
        plt.xlim([xlim_left, xlim_right])
        plt.show()


    def visualize_tree(self):
        feature_names = self.features.copy()
        feature_names.remove('energyoutput')
        feature_names.remove('timeepoch')
        export_graphviz(self.model, out_file='tree.dot',  
            feature_names=feature_names)
        # graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
        # Image(graph.create_png())


    def plot_histogram_feature_importances(self):
        feature_importance = self.model.feature_importances_
        feature_names = self.features.copy()
        feature_names.remove('energyoutput')
        feature_names.remove('timeepoch')
        feature_names = np.array(feature_names)
        feature_indices = np.arange(len(self.features) - 2)
        # make importances relative to max importance
        # feature_importance = 100.0 * (feature_importance / feature_importance.max())
        plt.bar(feature_indices, feature_importance, align='center', alpha=0.5)
        plt.xticks(feature_indices, feature_names, rotation=90)
        plt.xlabel('Features')
        plt.ylabel('Frequency of Feature Selection')
        plt.title('Decision Tree Regressor Feature Selection Frequency')
        plt.show()


    def histogram(self):

        # convert GridSearchCV class to a decision tree class
        if isinstance(self.model, GridSearchCV):
            self.model = self.model.best_estimator_
            
        tree = self.model.tree_

        def count_feature_occurrences(node, feature_counts):
            if tree.feature[node] != -2:
                feature_idx = tree.feature[node]
                if feature_idx not in feature_counts:
                    feature_counts[feature_idx] = 1
                else:
                    feature_counts[feature_idx] += 1
                count_feature_occurrences(tree.children_left[node], feature_counts)
                count_feature_occurrences(tree.children_right[node], feature_counts)

        # create a dictionary to store the feature counts
        feature_counts = dict()

        # traverse the tree and count the occurrences of each feature
        count_feature_occurrences(0, feature_counts)

        feature_names = self.features.copy()
        feature_names.remove('energyoutput')
        feature_names.remove('timeepoch')
        # display the results
        keys = list(feature_counts.keys())
        values = list(feature_counts.values())
        # sort the feature_names based on the keys
        feature_names = [feature_names[i] for i in keys]                
        plt.bar(range(len(feature_counts)), values, align='center', alpha=0.5)
        plt.xticks(range(len(feature_counts)), feature_names, rotation=65)
        plt.ylabel('Quantity of Feature Selection')
        plt.title('Decision Tree Regressor Feature Selection Quantity', y=1.05)
        plt.rcParams.update({'font.size': 14})
        plt.show()

    
    def visualize_3d_plot(self):
        energy = self.y_test
        radiation = self.X_test[:, 0]
        temperature = self.X_test[:, 1]
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'surface'}]])
        fig.add_trace(go.Surface(x=radiation , y=temperature, z=energy, \
            showscale=True, opacity=0.5), row=1, col=1)

        fig.add_trace(go.Scatter3d(x=radiation, y=temperature, z=energy, mode='markers', \
            marker=dict(size=5, color='red')), row=1, col=1)
        
        fig.update_layout(title=f'Solarsystem id: {self.solarsystem_id}, Location: {self.location}', \
            scene = dict(
                xaxis_title='Solar radiation',
                yaxis_title='Temperature',
                zaxis_title='Energy output',
                xaxis = dict(
                    backgroundcolor="rgb(200, 200, 230)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white",),
                yaxis = dict(
                    backgroundcolor="rgb(230, 200,230)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white"),
                zaxis = dict(
                    backgroundcolor="rgb(230, 230,200)",
                    gridcolor="white",
                    showbackground=True,
                    zerolinecolor="white",),),
                # width=700, height=700, 
                autosize=True, 
                # margin=dict(l=65, r=50, b=65, t=90)
                )
        
        fig.show()
        


    