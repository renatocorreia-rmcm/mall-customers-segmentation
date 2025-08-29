"""
the distance between 2 vectors is the scalar ||v1-v2||
so the variance in each cluster is  somatory ( ||centroid-vi||**2 )
"""


def printlog(string: str, print_log: bool):
    if print_log:
        print(string)


def euclidian_distance(centroid, point):
    """

    :param centroid:
    :param point:
    :return: Euclidean distance between centroid and point
    """

    if centroid.ndim == 1:  # distance from point to single centroid
        return ((centroid - point) ** 2).sum(axis=0)

    if centroid.ndim == 2:  # distance from point to multiple centroids - return array with pair-wise distances
        return ((centroid - point) ** 2).sum(axis=1)


def variance(clusters, centroids):  # somatory (variance of clusters)
    v = 0
    k = len(clusters)
    for i in range(k):
        v += variance_cluster(clusters[i], centroids[i])
    return v


def variance_cluster(cluster, centroid):  # somatory (||centroid-vi||**2) = somatory (euclidian_distance(centroid, vi)**2)
    v = 0
    for vector in cluster:
        v += euclidian_distance(centroid, vector)**2
    return v
