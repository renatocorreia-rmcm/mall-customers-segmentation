"""
the distance between 2 vectors is the scalar ||v1-v2||
so the variance in each cluster is  somatory ( ||centroid-vi||**2 )
"""


def euclidian_distances(centroids, point):  # todo: merge with euclidian_distance()
    """

    :param centroids:
    :param point:
    :return: array of Euclidean distances from centroid_i to point
    """

    return ((centroids-point)**2).sum(axis=1)


def euclidian_distance(centroid, point):  # todo: merge with euclidian_distances()
    """

    :param centroid:
    :param point:
    :return: Euclidean distance between centroid and point
    """

    return ((centroid - point) ** 2).sum()


def variance(clusters, centroids):  # somatory (variance of clusters)
    v = 0
    k = len(clusters)
    for i in range(k):
        v += variance_cluster(clusters[i], centroids[i])
    return v


def variance_cluster(cluster, centroid):  # somatory (||centroid-vi||**2) = somatory (euclidian_distance(centroid, vi)**2)
    v = 0
    # todo: get cluster. recalculate? pass through functions?
    for vector in cluster:
        v += euclidian_distance(centroid, vector)**2
    return v
