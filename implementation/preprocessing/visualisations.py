import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random
from matplotlib.colors import ListedColormap,colorConverter
from sklearn.decomposition import PCA,RandomizedPCA,NMF
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D


def make_heat_map_pca(X,y):
    pca = PCA(n_components=2)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca.fit(X_scaled)
    feature_names = X.columns.values.tolist()
    plt.close()
    plt.matshow(pca.components_, cmap='viridis')
    plt.yticks([0,1],["First component","Second component"])
    plt.colorbar()
    plt.xticks([x+10 for x in range(len(feature_names))], feature_names, rotation=60, ha='left')
    plt.xlabel("Features")
    plt.ylabel("Principal components")
    plt.savefig("./visualisations/heat_map_pca.png",format='png',dpi=500)
    plt.show()


def make_scatter_plot(X,y):
    viz_name = "./visualisations/"
    """
    Principal Component Analysis

    pca = PCA(n_components=2,svd_solver='randomized',random_state=5)
    scaler = MinMaxScaler() #StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_trans = pca.fit(X_scaled).transform(X_scaled)
    viz_name += "scatter_pca.png"
    """

    """
    NMF: Non-Negative Matrix Factorization
    - notes:
        Not any better than PCA

    nmf = NMF(n_components=2, random_state=10)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_trans = nmf.fit_transform(X_scaled)
    viz_name += "scatter_nmf.png"
    """

    """
    **Manifold Learning**
    Manifold learning algorithms are useful for visualisations,they create more complex mappings,
    they compute a new representation once only and are not good as learning algorithms
    __t-SNE__ : A Manifold learning algorithm
    """
    tsne = TSNE(n_components=2,random_state=random.randint(1,50))
    scaler = MaxAbsScaler()
    X = scaler.fit_transform(X)
    X_trans = tsne.fit_transform(X)
    viz_name += "scatter_tsne.png"

    targets = list(set(y.values.tolist()))
    targets = [1 if x == 'p' else 0 for x in targets]
    p_xs = np.where(y == 'p')[0]
    n_xs = np.where(y == 'n')[0]

    plt.close()
    print("Plotting scatter graph for data X and y ...")
    plt.figure(figsize=(8,8))
    plt.legend(['Pass','Not Passing'],loc='best')
    plt.xlabel("First principal component")
    plt.ylabel("Second principal component")

    for rng,color,marker in [(p_xs,'red','^'),(n_xs,'blue','o')]:
        plt.scatter(X_trans[rng,0],X_trans[rng,1],c=color,marker=marker)
    plt.savefig(viz_name,format='png',dpi=500)
    plt.show()

def make_scatter_plot_3D(X,y):
    # pca = PCA(n_components=3,svd_solver='randomized',random_state=5)
    # scaler = MaxAbsScaler() #StandardScaler()
    # X_scaled = scaler.fit_transform(X)
    # X_trans = pca.fit(X_scaled).transform(X_scaled)

    # nmf = NMF(n_components=3, random_state=10)
    # scaler = MinMaxScaler()
    # X_scaled = scaler.fit_transform(X)
    # X_pca = nmf.fit_transform(X_scaled)


    tsne = TSNE(n_components=3,random_state=random.randint(1,50))
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    X_trans = tsne.fit_transform(X)

    p_xs = np.where(y == 'p')[0]
    n_xs = np.where(y == 'n')[0]

    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    print("Plotting 3D scatter graph for data X and y ...")
    for rng,color,marker in [(p_xs,'red','^'),(n_xs,'blue','o')]:
        ax.scatter(X_trans[rng,0],X_trans[rng,1],X_trans[rng,2],c=color,marker=marker)
    plt.show()


def make_viz_per_feature_histogram(X,y):
    #TODO: Needs fixing
    num_rows, num_features = X.shape
    rows = ( num_features//7 ) + (num_features % 7)
    fig, axes = plt.subplots(rows,7,figsize=(10,20))

    p_xs = np.where(y == 'p')[0]
    n_xs = np.where(y == 'n')[0]
    pass_xs = X.iloc[p_xs]
    not_pass_xs = X.iloc[n_xs]
    cmap = ListedColormap(['#0000aa', '#ff2020', '#50ff50'])

    ax = axes.ravel()
    for i in range(num_features):
        print("Plotting histogram for feature {}".format(i+1))
        _, bins = np.histogram(X.iloc[:,[i]], bins=50)
        passes = pass_xs.iloc[:,[i]].values.tolist()
        fails = not_pass_xs.iloc[:,[i]].values.tolist()
        ax[i].hist(passes,bins=bins,alpha=0.5)
        ax[i].hist(fails,bins=bins,alpha=0.5)
        ax[i].set_title(X.columns.values.tolist()[i])
        ax[i].set_yticks(())
    ax[0].set_xlabel("Feature magnitude")
    ax[0].set_ylabel("Frequency")
    ax[0].legend(["Pass","Not Pass"],loc="best")
    fig.tight_layout()
    plt.savefig("./visualisations/feature_hist.png",format='png',dpi=500)
    plt.show()





def discrete_scatter(x1, x2, y=None, markers=None, s=10, ax=None,
                     labels=None, padding=.2, alpha=1, c=None, markeredgewidth=None):
    #TODO: Discard and use your scatter plotting function
    """Adaption of matplotlib.pyplot.scatter to plot classes or clusters.

    Parameters
    ----------

    x1 : nd-array
        input data, first axis

    x2 : nd-array
        input data, second axis

    y : nd-array
        input data, discrete labels

    cmap : colormap
        Colormap to use.

    markers : list of string
        List of markers to use, or None (which defaults to 'o').

    s : int or float
        Size of the marker

    padding : float
        Fraction of the dataset range to use for padding the axes.

    alpha : float
        Alpha value for all points.
    """
    if ax is None:
        ax = plt.gca()

    if y is None:
        y = np.zeros(len(x1))

    unique_y = np.unique(y)

    if markers is None:
        markers = ['o', '^', 'v', 'D', 's', '*', 'p', 'h', 'H', '8', '<', '>'] * 10

    if len(markers) == 1:
        markers = markers * len(unique_y)

    if labels is None:
        labels = unique_y

    # lines in the matplotlib sense, not actual lines
    lines = []

    current_cycler = mpl.rcParams['axes.prop_cycle']

    for i, (yy, cycle) in enumerate(zip(unique_y, current_cycler())):
        mask = y == yy
        # if c is none, use color cycle
        if c is None:
            color = cycle['color']
        elif len(c) > 1:
            color = c[i]
        else:
            color = c
        # use light edge for dark markers
        if np.mean(colorConverter.to_rgb(color)) < .4:
            markeredgecolor = "grey"
        else:
            markeredgecolor = "black"

        lines.append(ax.plot(x1[mask], x2[mask], markers[i], markersize=s,
                             label=labels[i], alpha=alpha, c=color,
                             markeredgewidth=markeredgewidth,
                             markeredgecolor=markeredgecolor)[0])

    if padding != 0:
        pad1 = x1.std() * padding
        pad2 = x2.std() * padding
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.set_xlim(min(x1.min() - pad1, xlim[0]), max(x1.max() + pad1, xlim[1]))
        ax.set_ylim(min(x2.min() - pad2, ylim[0]), max(x2.max() + pad2, ylim[1]))

    return lines
