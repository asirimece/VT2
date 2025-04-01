# methods.py
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

def kmeans_clustering(X, **params):
    """
    Performs K-Means clustering on X using parameters provided in params.
    """
    kmeans = KMeans(**params)
    labels = kmeans.fit_predict(X)
    return labels, kmeans

def hierarchical_clustering(X, **params):
    """
    Performs hierarchical clustering on X using parameters provided in params.
    """
    hc = AgglomerativeClustering(**params)
    labels = hc.fit_predict(X)
    return labels, hc

def dbscan_clustering(X, **params):
    """
    Performs DBSCAN clustering on X using parameters provided in params.
    """
    dbscan = DBSCAN(**params)
    labels = dbscan.fit_predict(X)
    return labels, dbscan
