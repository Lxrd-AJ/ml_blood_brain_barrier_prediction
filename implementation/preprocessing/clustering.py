from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from collections import Counter
from sklearn.decomposition import PCA,RandomizedPCA,NMF
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from .visualisations import discrete_scatter
import matplotlib.pyplot as plt
import numpy as np

"""
Performs poorly using DBSCAN algorithm
def dbscan(X,y,viz_name,should_scale):
    dbsc = DBSCAN()
    if should_scale:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X = scaler.fit_transform(X)
    clusters = dbsc.fit_predict(X)
    print("Original data in datset {}".format(Counter(y)))
    print("Cluster members:\n{}".format(Counter(clusters)))
"""

def make_agglomerative_cluster_plot(X,y,should_scale,viz_name):
    agg_clt = AgglomerativeClustering(n_clusters=2)
    if should_scale:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    clusters = agg_clt.fit_predict(X)
    print("Original data in datset {}".format(Counter(y)))
    print("Clusters found {}".format(Counter(clusters)))
    plt.close()
    discrete_scatter(X[:,0],X[:,1],clusters)
    plt.legend(loc=2)
    plt.xlabel("First Principal component")
    plt.ylabel("Second Principal component")
    plt.savefig(viz_name,format='png',dpi=500)
    plt.show()

def make_cluster_plot(X,y,viz_name):
    #TODO: Try scaling X using the MinMaxScaler
    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)
    assignments = kmeans.labels_
    classes = np.unique(assignments)

    print("Original data in datset {}".format(Counter(y)))
    print("Clustering data found in X {}".format(Counter(kmeans.labels_)))

    fig, axes = plt.subplots(1,2,figsize=(10,5))

    p_xs = np.where(y == 'p')[0]
    n_xs = np.where(y == 'n')[0]
    for rng,color,marker in [(p_xs,'red','^'),(n_xs,'blue','o')]:
        axes[1].scatter(X[rng,0],X[rng,1],c=color,marker=marker)

    for x,marker,clr in zip(classes,'^o',['red','blue']):
        x_idx = np.where(assignments == x)[0]
        axes[0].scatter(X[x_idx,0],X[x_idx,1],c=clr,marker=marker)
        axes[0].scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],marker='^',
                        edgecolor='green',linewidth='3',s=100,facecolor='yellow')

    plt.savefig(viz_name,format='png',dpi=500)
    # plt.show()
