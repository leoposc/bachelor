from db import DataManager 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


class ScikitManager():
    X_df : pd.DataFrame
    y_df : pd.DataFrame
    X_train : np.ndarray
    y_train : np.ndarray
    X_test : np.ndarray
    y_test : np.ndarray 
    location: str
    solarsystem_id: int
    start: str
    end: str


    def __init__(self, location: str, solarsystem_id: int, start: str, end: str):
        self.location = location
        self.solarsystem_id = solarsystem_id
        self.start = start
        self.end = end


    def fetch_data(self):
        dm = DataManager()
        dm.fetch_weather_data(self.location, self.start, self.end)
        dm.fetch_solar_data(self.location, self.solarsystem_id, self.start, self.end)


    def get_data(self):
        dm = DataManager()
        solar_df = dm.select_solar_data(self.location, self.solarsystem_id)
        weather_df = dm.select_weather_data(self.location)
        # merge data on timeepoch
        data_df = weather_df.merge(solar_df, on='timeepoch')        

        self.XY_df   = data_df
        self.X_df    = weather_df
        self.y_df    = solar_df
        self.X_train = data_df.iloc[:, 1:-1].values
        self.y_train = data_df.iloc[:, -1].values.reshape(-1,1)
        # warning: check the diminsions of y


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


    def visualize_pairwise_correlation(self):
        cols = ['energyoutput', 'solarradiation', 'hour', 'temperature', 'wind', 'cloudcoverage']
        sns.pairplot(self.XY_df[cols], size=2.5)
        plt.tight_layout()
        plt.show()


    def visualize_heatmap(self):
        cols = ['energyoutput', 'solarradiation', 'hour', 'temperature', 'wind', 'cloudcoverage']
        cm = np.corrcoef(self.XY_df[cols].values.T)
        sns.set(font_scale=1.5)
        hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
        plt.show()

    
    

    

