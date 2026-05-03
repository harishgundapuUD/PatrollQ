import os
import json
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler


class DataCleaner:
    def __init__(self, file_path, output_path="", config_file=""):
        self.file_path = file_path
        self.output_path = output_path
        self.config_file = config_file
        self.config = {}

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        if self.config_file and os.path.exists(self.config_file):
            with open(self.config_file, 'r') as f:
                self.config = json.load(f)
    
    def remove_missing_values(self):
        self.df.dropna(inplace=True)
    
    def save_cleaned_data(self):
        if self.output_path:
            self.df.to_csv(self.output_path, index=False)


class FeatureEngineering:
    def __init__(self, df, output_path="", config={}):
        self.df = df
        self.output_path = output_path
        self.config = config

    def convert_to_datetime(self, cols):
        for col in cols:
            self.df[col] = pd.to_datetime(self.df[col])
    
    def extract_date_time_features(self, cols):
        for col in cols:
            self.df[f'{col}_date'] = self.df[col].dt.date
            self.df[f'{col}_time'] = self.df[col].dt.time
            self.df[f'{col}_hour'] = self.df[col].dt.hour
            self.df[f'{col}_minute'] = self.df[col].dt.minute
            self.df[f'{col}_day'] = self.df[col].dt.day
            self.df[f'{col}_month'] = self.df[col].dt.month
            self.df[f'{col}_year'] = self.df[col].dt.year
            self.df[f'{col}_dayofweek'] = self.df[col].dt.dayofweek  # 0=Monday
    
    def drop_original_datetime_cols(self, cols):
        self.df.drop(columns=cols, inplace=True)

    def create_crime_category(self):
        if self.config:
            self.df['Crime Group'] = (
                                        self.df['Primary Type']
                                        .str.strip()
                                        .str.upper()
                                        .map(self.config.get('crime_group_map', {}))
                                        .fillna('Other')
                                    )
    
    def create_season_feature(self):
        season_map = self.config.get("season_map", {})

        # convert JSON → reverse mapping (month → season)
        month_to_season = {
                                month: season
                                for season, months in season_map.items()
                                for month in months
                            }

        self.df['Season'] = self.df['Date_month'].map(month_to_season).fillna("Unknown")
    
    def create_geospatial_features(self, lat_col='Latitude', lon_col='Longitude'):
        geo_config = self.config.get('geo_features', {})
        if not geo_config:
            print("No geo features config found.")
            return

        # Binning
        lat_bins = geo_config.get('Latitude', {}).get('bins', 10)
        lon_bins = geo_config.get('Longitude', {}).get('bins', 10)

        self.df['Lat_bin'] = pd.cut(self.df[lat_col], bins=lat_bins, labels=False)
        self.df['Lon_bin'] = pd.cut(self.df[lon_col], bins=lon_bins, labels=False)
        self.df['Geo_Cluster'] = self.df['Lat_bin'].astype(str) + "_" + self.df['Lon_bin'].astype(str)

        # KMeans clustering
        n_clusters = geo_config.get('District_Clusters', 20)
        coords = self.df[[lat_col, lon_col]].dropna()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.df.loc[coords.index, 'District_Cluster'] = kmeans.fit_predict(coords)
    
    def create_crime_severity(self):
        severity_map = self.config.get('crime_severity', {})
        if not severity_map:
            print("No crime severity mapping found in config.")
            return
        
        # Map Crime Group to severity score
        self.df['Crime_Severity'] = self.df['Crime Group'].map(severity_map).fillna(1)
    
    def location_feature_engineering(self):
        location_group_map = self.config.get('location_group_map', {})
        self.df['Location_Group'] = self.df['Location Description'].map(location_group_map).fillna('Other')
    
    def normalize_coordinates(self, method='minmax'):
        if method == 'minmax':
            scaler = MinMaxScaler()
            df_scaled = scaler.fit_transform(self.df[['Latitude', 'Longitude']])
            self.df['Latitude_norm'] = df_scaled[:, 0]
            self.df['Longitude_norm'] = df_scaled[:, 1]
        elif method == 'standard':
            self.df['Latitude_norm'] = (self.df['Latitude'] - self.df['Latitude'].mean()) / self.df['Latitude'].std()
            self.df['Longitude_norm'] = (self.df['Longitude'] - self.df['Longitude'].mean()) / self.df['Longitude'].std()
    
    def create_engineered_features(self, cols):
        self.convert_to_datetime(cols=cols)
        self.extract_date_time_features(cols=cols)
        self.drop_original_datetime_cols(cols=cols)
        self.create_crime_category()
        self.create_season_feature()
        self.create_geospatial_features()
        self.create_crime_severity()
        self.location_feature_engineering()
        self.normalize_coordinates()

    def save_engineered_data(self):
        if self.output_path:
            self.df.to_csv(self.output_path, index=False)


data_cleaner = DataCleaner(file_path="dataset/crimes.csv", config_file="utils/config.json")
data_cleaner.load_data()
data_cleaner.remove_missing_values()
feature_engineer = FeatureEngineering(df=data_cleaner.df, config=data_cleaner.config, output_path="dataset/cleaned_crimes.csv")
feature_engineer.create_engineered_features(cols=['Date', 'Updated On'])
feature_engineer.save_engineered_data()