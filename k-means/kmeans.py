import numpy as np
import random
from modules import *


def aux_kmeans(df, k: int, get_dist):
    """
    :df: dataframe representing dataset
    :k: amount of clusters
    :get_dist: function to calculate distance from datapoint to each centroid. must return array of distances from point to centroid_i

    runs a single instance of kmeans (1 initialization)

    :return: arrays of k centroids coordinates
    """

    """ initialize datapoints """  # as array of vectors
    datapoints = df.iloc[:].values  # each datapoint is a 4D vector representing a costumer

    """ initialize centroids """  # as random existing datapoints
    initial_centroids_i = np.array(random.sample(range(len(datapoints)), k))  # start with k random clusters represented by index of existing customer
    centroids = datapoints[initial_centroids_i]

    """"""
    clusters = []  # contains the array of vectors of each cluster
    for i in range(k):
        clusters.append([])
    for i in range(30):  # todo: stop when detect saturation
        """ ASSIGN each datapoint to a cluster """
        for point in datapoints:
            distances = get_dist(centroids, point)
            i_cluster = np.argmin(distances)
            clusters[i_cluster].append(point)
        """ FIT each cluster to its datapoints mean """  # todo: the correct would be to find the position wich minimizes get_dist()  # for euclidian distance, this actually is the mean position
        for i_cluster in range(k):
            centroids[i_cluster] = np.sum(clusters[i_cluster], axis=0)/len(clusters[i_cluster])

    return centroids, clusters


def kmeans(df, k: int = -1, get_dist=euclidian_distances, initializations: int = 15):
    """

    :param df: pandas dataframe representing dataset
    :param k: amount of clusters
    :param get_dist: vectorial distance function
    :param initializations: how many instances of k-means to run before picking the best solution one
    :return: array with centroids coordinates (k, 1) and array with clusters contents (k, len(cluster_i))
    """

    """ optmize k """
    if k == -1:  # todo: compute best k and use it
        pass

    """ minimum solution """
    min_solution_variance = float('inf')
    min_centroids = []
    min_clusters = []  # todo: solve: unused

    for i in range(initializations):
        # compute new solution
        centroids, clusters = aux_kmeans(df, k, get_dist)
        solution_variance = variance(clusters, centroids)
        # compare to current best solution
        if solution_variance < min_solution_variance:
            min_solution_variance = solution_variance
            min_centroids = centroids
            min_clusters = clusters

    print(f"Total variance of best solution found: {min_solution_variance}")

    """ assign clusters """

    points_clusters = np.full(shape=200, fill_value="", dtype=object)  # array with the cluster of each point
    for i, point in enumerate(df.iloc[:].values):
        cluster_of_point = np.argmin(get_dist(min_centroids, point))
        points_clusters[i] = cluster_of_point

    return min_centroids, points_clusters
