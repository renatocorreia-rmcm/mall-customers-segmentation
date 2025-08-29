import numpy as np
import random
from modules import *


def aux_kmeans(df, k: int, print_log: bool):
    """
    :df: dataframe representing dataset
    :k: amount of clusters

    runs a single instance of kmeans (1 initialization)

    :return: arrays of k centroids coordinates
    """

    """ initialize datapoints """  # as array of vectors
    datapoints = df.iloc[:].values  # each datapoint is a 4D vector representing a costumer

    """ initialize centroids """  # as random existing datapoints
    initial_centroids_i = np.array(random.sample(range(len(datapoints)), k))  # start with k random clusters represented by index of existing customer
    centroids = datapoints[initial_centroids_i]

    """ repeat ASSIGN and FIT until saturation """
    # pseudo initializations
    previous_variance = 0
    current_variance = 1

    clusters = []

    printlog(f"\n\nSTARTING INSTANCE OF KMEANS with random centroids\n", print_log)
    printlog(f"Variance in each ASSIGN - FIT cycle\n", print_log)

    while current_variance != previous_variance:
        clusters = []  # contains the array of vectors of each cluster
        for i in range(k):
            clusters.append([])
        """ ASSIGN each datapoint to a cluster """
        for point in datapoints:
            distances = euclidian_distance(centroids, point)
            i_cluster = np.argmin(distances)
            clusters[i_cluster].append(point)
        """ FIT each cluster to its datapoints mean """
        for i_cluster in range(k):
            centroids[i_cluster] = np.sum(clusters[i_cluster], axis=0)/len(clusters[i_cluster])
        """ update variation """
        previous_variance = current_variance
        current_variance = variance(clusters, centroids)

        printlog(f"{current_variance}", print_log)
    printlog("(saturation)", print_log)
    printlog(f"\nthis instance converged to a local minimum with variance: {current_variance}", print_log)

    return centroids, clusters, current_variance


def kmeans(df, k: int = -1, initializations: int = 15, print_log: bool = False):
    """

    :param df: pandas dataframe representing dataset
    :param k: amount of clusters
    :param initializations: how many instances of k-means to run before picking the best solution one
    :param print_log: print log of execution
    :return: array with centroids coordinates (k, 1) and array with clusters contents (k, len(cluster_i))
    """

    """ optmize k """
    if k == -1:  # todo: how to pick k?
        pass

    """ find best solution """
    min_solution_variance = float('inf')
    min_centroids = []
    min_clusters = []  # todo: solve: unused

    for i in range(initializations):
        # compute new solution
        centroids, clusters, solution_variance = aux_kmeans(df, k, print_log)
        # compare to current best solution
        if solution_variance < min_solution_variance:
            min_solution_variance = solution_variance
            min_centroids = centroids
            min_clusters = clusters

    printlog(f"\n\nTotal variance of best solution found: {min_solution_variance}\n\n", print_log)

    """ assign clusters """
    points_clusters = np.full(shape=df.shape[0], fill_value="", dtype=object)  # array with the cluster of each point
    for i, point in enumerate(df.iloc[:].values):
        cluster_of_point = np.argmin(euclidian_distance(min_centroids, point))
        points_clusters[i] = cluster_of_point

    return min_centroids, points_clusters
