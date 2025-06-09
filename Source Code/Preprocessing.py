# Libraries
import numpy as np
import pandas as pd
from datetime import datetime # to get the current datetime

import matplotlib.pyplot as plt # to make plots (histograms, etc...)
import seaborn as sns
import plotly.express as px

import ast # transform the strings into a list

from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator # to automatically find the "knee" for the dbscan's parameter
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA



def log_transform_lifetime_features(dataframe):
    cols_to_transform = [col for col in dataframe.columns if col.startswith('lifetime_')]
    dataframe[cols_to_transform] = dataframe[cols_to_transform].apply(lambda x: np.log1p(x))
    return dataframe


customer_info = pd.read_csv('../data/raw/customer_info.csv', 
                            index_col=["customer_id"]).drop(["Unnamed: 0", 
                                                             "customer_name"],
                                                               axis='columns')

def preprocess_customer_data(df):
    # FEATURE SELECTION (Pt.1)

    # Altering the customer_birthdate column to datetime format
    df["customer_birthdate"] = pd.to_datetime(df["customer_birthdate"], 
                                                        format="%m/%d/%Y %I:%M %p")

    # Creating the age column
    today = datetime.now()
    df["age"] = df["customer_birthdate"].apply(lambda x: today.year - x.year - ((today.month, today.day) < (x.month, x.day))) # works because True = 1 and False = 0

    # Dropping birthdate column
    df = df.drop(columns=["customer_birthdate"])

    # Label encoding
    df["customer_gender"] = np.where(df["customer_gender"] == "female", 0, 1)

    # Converting Loyalty card to a binary variable 
    df['loyalty_card_number'] = df['loyalty_card_number'].notna().astype(int) 
    # notna() returns True if the value is not NaN, and False otherwise. We convert it to int to get 1 for True and 0 for False.

    # DATA CLEANING

    # Imputing missing values
    imputer = KNNImputer(n_neighbors=10)
    df[:] = imputer.fit_transform(df)

    # replacing  the percentage_of_products_bought_promotion below 0 with 0
    df["percentage_of_products_bought_promotion"] = np.where(
        df["percentage_of_products_bought_promotion"] < 0,
        0,
        df["percentage_of_products_bought_promotion"]
    )

    non_scaled_non_missing_customer_info = df # we will need to export this variable later on

    # FEATURE SELECTION (Pt.2)
    df = df.drop(["year_first_transaction", 
                                        "loyalty_card_number", 
                                        "latitude", 
                                        "longitude"], 
                                        axis=1)

    # Applying log transformation
    df = log_transform_lifetime_features(df)

    # Scaling
    scaler = RobustScaler()
    scaled_array = scaler.fit_transform(df)
    scaled_customer_info = pd.DataFrame(scaled_array, 
                                        columns=df.columns, 
                                        index=df.index)
    df = scaled_customer_info


    # Multidimentional Outliers
    db = DBSCAN(eps=2.5, min_samples=20)
    db.fit(df)

    labels = db.labels_ 
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0) 
    n_outliers = list(labels).count(-1)

    # removing the specifical rows that correspond to outliers
    df['cluster'] = labels
    df = df[df['cluster'] != -1]

    # dropping the cluster column 
    df = df.drop('cluster', axis=1)

    # removing the specifical rows that correspond to outliers
    non_scaled_non_missing_customer_info['cluster'] = labels
    non_scaled_non_missing_customer_info = non_scaled_non_missing_customer_info[non_scaled_non_missing_customer_info['cluster'] != -1]

    # dropping the cluster column 
    non_scaled_non_missing_customer_info = non_scaled_non_missing_customer_info.drop('cluster', axis=1)
    export_path = '../data/raw/non_scaled_non_missing_customer_info.csv'
    non_scaled_non_missing_customer_info.to_csv(export_path, index=True)


    # Exporting clean dataset
    export_path = '../data/raw/cleanerer_customer_info.csv'
    df.to_csv(export_path, index=True)




preprocess_customer_data(customer_info)