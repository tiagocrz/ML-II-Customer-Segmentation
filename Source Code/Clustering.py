import pandas as pd
import numpy as np

from sklearn.cluster import AgglomerativeClustering
import umap.umap_ as umap
from sklearn.cluster import KMeans



clustering_customer_info = pd.read_csv('../data/raw/cleanerer_customer_info.csv', 
                                       index_col="customer_id").drop(columns=[
                                           'customer_gender',
                                            'age',
                                            'percentage_of_products_bought_promotion',
                                            'typical_hour',
                                            'distinct_stores_visited',
                                            'number_complaints',
                                            "kids_home",
                                            "teens_home"])

def best_sol():
    n_clusters = 7

    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    hierarchical_labels = hierarchical.fit_predict(clustering_customer_info)

    reducer = umap.UMAP(random_state=42)
    embedding_hierarchical = reducer.fit_transform(clustering_customer_info)

    # using the hierarchical cluster centers as initialization for KMeans
    # To get centers, we calculate the mean of each hierarchical cluster
    cluster_centers = np.array([clustering_customer_info[hierarchical_labels == i].mean(axis=0) for i in range(n_clusters)])

    # run KMeans initialized with  the hierarchical centers
    kmeans = KMeans(n_clusters=n_clusters, init=cluster_centers, n_init=1, random_state=42)
    kmeans_labels = kmeans.fit_predict(clustering_customer_info)
    
    return kmeans_labels