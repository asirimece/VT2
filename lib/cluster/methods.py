# methods.py
import os
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

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
    params['eps'] = float(params['eps'])
    dbscan = DBSCAN(**params)
    labels = dbscan.fit_predict(X)
    return labels, dbscan

def plot_pca_scree_plot(X, out_dir="cluster_plots"):
    """
    Generates a scree plot showing eigenvalues vs. principal component number.
    
    Parameters:
        X (ndarray): The data matrix, each row is a sample.
        out_dir (str): Folder path to save the scree plot.
    
    The number of PCA components is set to min(n_samples, n_features) to avoid errors.
    """
    # Ensure the output folder exists
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    n_samples, n_features = X.shape
    n_components = min(n_samples, n_features)  # maximum valid number of components
    pca = PCA(n_components=n_components)
    pca.fit(X)

    # explained_variance_ is the array of eigenvalues for each principal component.
    eigenvalues = pca.explained_variance_
    components = range(1, len(eigenvalues) + 1)

    plt.figure()
    plt.plot(components, eigenvalues, marker='o')
    plt.title("Scree Plot")
    plt.xlabel("Principal Component Number")
    plt.ylabel("Eigenvalue")
    
    scree_plot_path = os.path.join(out_dir, "scree_plot.png")
    plt.savefig(scree_plot_path)
    plt.close()
    print(f"[INFO] Scree plot saved to {scree_plot_path}")

def evaluate_k_means(X, base_params, k_values):
    """
    Evaluate KMeans clustering over a range of k values.
    For each k, compute the WCSS (inertia) and silhouette score.
    In addition, produce a PCA scree plot (eigenvalue vs. principal component number)
    by calling plot_pca_scree_plot().
    
    Parameters:
        X (ndarray): Data matrix where each row is a sample.
        base_params (dict): Base KMeans parameters from config (except n_clusters).
        k_values (list): List of k values to test.
    
    Returns:
        inertias (list): The within-cluster sum of squares for each k.
        silhouettes (list): The silhouette score for each k (None if not computed).
    """
    inertias = []
    silhouettes = []
    n_samples = X.shape[0]

    # Evaluate K-Means for each k
    for k in k_values:
        params = base_params.copy()
        params['n_clusters'] = k
        kmeans = KMeans(**params)
        labels = kmeans.fit_predict(X)
        inertia = kmeans.inertia_
        inertias.append(inertia)

        # Compute silhouette score when valid
        if 1 < k < n_samples:
            sil_score = silhouette_score(X, labels)
        else:
            sil_score = None
        silhouettes.append(sil_score)

        print(f"k = {k}: Inertia = {inertia:.6e}, Silhouette Score = {sil_score}")

    # Generate the PCA scree plot
    plot_pca_scree_plot(X, out_dir="cluster_plots")

    return inertias, silhouettes