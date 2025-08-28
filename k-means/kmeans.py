import pandas as pd
import numpy as np
import random

def dist(centroids, point):
    """
    :return: array of distance from centroid_i to point
    """

    return ((centroids-point)**2).sum(axis=1)


def kmeans(df, k: int):
    """
    :df: dataframe representing dataset

    :return: arrays of k centroids
    """
    
    """ initialize datapoints (vectors) """
    datapoints = df.iloc[:].values  # each datapoint is a customer
    amount_vectors = datapoints.shape[0]
    """ initialize centroids """
    initial_centroids_i = np.array(random.sample(range(amount_vectors), k))  # start with k random clusters represented by index of existing customer
    centroids = datapoints[initial_centroids_i]
    clusters = []
    for i in range(k): clusters.append([]) 

    for i in range(30):  # TO-DO: STOP ONLY WHEN DETECT SATURATION
        """ get cluster of each vector """  # ASSIGN
        for point in datapoints:
            distances = dist(centroids, point)
            i_cluster = np.argmin(distances)
            clusters[i_cluster].append(point)
        
        """ re-calculate center of each cluster """  # FIT
        for i_cluster in range(k):
            centroids[i_cluster] = np.sum(clusters[i_cluster], axis=0)/len(clusters[i_cluster])
        

    
    return centroids