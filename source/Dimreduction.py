

from time import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble, lda,
                     random_projection)

def plot_embedding(X, y, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(y[i]),
                 color=plt.cm.Set1(y[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)
        
def dim_Reduction(data, label, opt, n_components, visualize = False):
    """Dimensionality Reduction

    Parameters
    ----------
    data:
        Feature table
    label:
        Class label of each data point (can be assigned randomly if not available)
    opt:
        Options for dimension reduction methods
    n_components:
        Number of components needed
    visualize:
        Visualize or not the dimension reduction result
    
    """

    ## Preparing training data---------------
    X = data
    y = label   

    n_samples = X.shape[0]    
    n_neighbors = 10

    if n_samples == X.size:
        print 'Only one data point'
        return X[0:n_components]
    
    ## Random 2d projection---------------
    if opt == 'rand':
    
        print("Computing random 2d projection")
        rp = random_projection.SparseRandomProjection(n_components, random_state=42)
        X_randProj = rp.fit_transform(X)
        X_projected = X_randProj
        if visualize == True:
            plot_embedding(X_projected, y, "Random Projection of the digits")

    ## PCA 2d projection---------------
    if opt == 'pca':
        print("Computing PCA projection")
        t0 = time()
        X_pca = decomposition.TruncatedSVD(n_components).fit_transform(X)
        X_projected = X_pca
        if visualize == True:
            plot_embedding(X_pca,y, 
                           "Principal Components projection of the digits (time %.2fs)" %
                           (time() - t0))

    ## linear discriminant 2d projection---------------
    if opt == 'lda':
        print("Computing LDA projection")
        X2 = X.copy()
        X2.flat[::X.shape[1] + 1] += 0.01  # Make X invertible
        t0 = time()
        X_lda = lda.LDA(n_components).fit_transform(X2, y)
        X_projected = X_lda
        if visualize == True:
            plot_embedding(X_lda,y, 
                   "Linear Discriminant projection of the digits (time %.2fs)" %
                   (time() - t0))
    

    ## Isomap projection---------------
    if opt == 'iso':
        print("Computing Isomap embedding")
        t0 = time()
        X_iso = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
        X_projected = X_iso
        if visualize == True:
            plot_embedding(X_iso,y, 
                   "Isomap projection of the digits (time %.2fs)" %
                   (time() - t0))

    ## Local linear Embedding projection---------------
    if opt == 'lle':
        print("Computing LLE embedding")
        clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                              method='standard')
        t0 = time()
        X_lle = clf.fit_transform(X)
        X_projected = X_lle
        print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
        if visualize == True:
            plot_embedding(X_lle,y, 
                       "Locally Linear Embedding of the digits (time %.2fs)" %
                       (time() - t0))

    ## Modified Locally linear embedding projection---------------
    if opt == 'mlle':
        print("Computing modified LLE embedding")
        clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                              method='modified')
        t0 = time()
        X_mlle = clf.fit_transform(X)
        X_projected = X_mlle
        print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
        if visualize == True:
            plot_embedding(X_mlle,y, 
                       "Modified Locally Linear Embedding of the digits (time %.2fs)" %
                       (time() - t0))

    ## HLLE embedding projection---------------
    if opt == 'HLLE':
        print("Computing Hessian LLE embedding")
        clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                              method='hessian')
        t0 = time()
        X_hlle = clf.fit_transform(X)
        X_projected = X_hlle
        print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
        if visualize == True:
            plot_embedding(X_hlle,y, 
                       "Hessian Locally Linear Embedding of the digits (time %.2fs)" %
                       (time() - t0))

    ## LTSA embedding projection---------------
    if opt == 'LTSA':
        print("Computing LTSA embedding")
        clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
                                              method='ltsa')
        t0 = time()
        X_ltsa = clf.fit_transform(X)
        X_projected = X_ltsa
        print("Done. Reconstruction error: %g" % clf.reconstruction_error_)
        if visualize == True:
            plot_embedding(X_ltsa,y, 
                       "Local Tangent Space Alignment of the digits (time %.2fs)" %
                       (time() - t0))

    ## MDS embedding projection---------------
    if opt == 'MDS':
        print("Computing MDS embedding")
        clf = manifold.MDS(n_components, n_init=1, max_iter=100)
        t0 = time()
        X_mds = clf.fit_transform(X)
        X_projected = X_mds
        print("Done. Stress: %f" % clf.stress_)
        if visualize == True:
            plot_embedding(X_mds,y, 
                       "MDS embedding of the digits (time %.2fs)" %
                       (time() - t0))

    ## Random Trees embedding projection---------------
    if opt == 'randtree':
        print("Computing Totally Random Trees embedding")
        hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,
                                               max_depth=5)
        t0 = time()
        X_transformed = hasher.fit_transform(X)
        pca = decomposition.TruncatedSVD(n_components)
        X_reduced = pca.fit_transform(X_transformed)
        X_projected = X_reduced
        if visualize == True:
            plot_embedding(X_reduced,y, 
                       "Random forest embedding of the digits (time %.2fs)" %
                       (time() - t0))
    
    ## Spectral embedding projection---------------
    if opt == 'spectral':
        print("Computing Spectral embedding")
        embedder = manifold.SpectralEmbedding(n_components, random_state=0,
                                              eigen_solver="arpack")
        t0 = time()
        X_se = embedder.fit_transform(X)
        X_projected = X_se
        if visualize == True:
            plot_embedding(X_se,y, 
                       "Spectral embedding of the digits (time %.2fs)" %
                       (time() - t0))
            
    if visualize == True:
        plt.show()

    return X_projected
