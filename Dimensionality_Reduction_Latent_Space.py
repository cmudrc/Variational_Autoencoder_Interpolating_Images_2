import numpy as np
import pacmap  # will need to change numba version: pip install numba==0.53
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


########################################################################################################################
# Latent Feature Cluster for Training Data using PaCMAP
def PaCMAP_reduction(latent_points, latent_dimensionality, random_state=1):
    # initializing the pacmap instance
    X = latent_points

    embedding = pacmap.PaCMAP(n_dims=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, random_state=random_state)

    # fit the data (The index of transformed data corresponds to the index of the original data)
    X_transformed = embedding.fit_transform(X, init="pca")

    # visualize the embedding
    x = X_transformed[:, 0]
    y = X_transformed[:, 1]
    title = "PaCMAP with Predicted Points\nLatent Space Dimensionality: " + str(latent_dimensionality)
    return x, y, title


########################################################################################################################
# Latent Feature Cluster for Training Data using PCA and Predicted Latent Points
def PCA_reduction(latent_points, latent_dimensionality, perplexity=7, learning_rate=20):
    pca = PCA(n_components=2)
    pca_fit = pca.fit_transform(latent_points)
    # configuring the parameters
    # the number of components = dimension of the embedded space
    # default perplexity = 30 " Perplexity balances the attention t-SNE gives to local and global aspects of the data.
    # It is roughly a guess of the number of close neighbors each point has. ..a denser dataset ... requires higher perplexity value"
    # default learning rate = 200 "If the learning rate is too high, the data may look like a ‘ball’ with any point
    # approximately equidistant from its nearest neighbours. If the learning rate is too low,
    # most points may look compressed in a dense cloud with few outliers."
    title = "PCA with Predicted Points \nPerplexity: " + str(perplexity) + "\nLearning Rate: " + str(
        learning_rate) + "\nLatent Space Dimensionality: " + str(latent_dimensionality)
    x = pca_fit[:, 0]
    y = pca_fit[:, 1]
    # predict = pca.fit_transform(latent_matrix)
    # predict_x = predict[:, 0]
    # predict_y = predict[:, 0]
    return x, y, title


########################################################################################################################
# Latent Feature Cluster for Training Data using PCA reduced T-SNE
def PCA_TSNE_reduction(latent_points, latent_dimensionality, perplexity=30, learning_rate=20):
    pca = PCA(n_components=2)
    pca_fit = pca.fit_transform(latent_points)
    model = TSNE(n_components=2, random_state=0, perplexity=perplexity,
                 learning_rate=learning_rate)  # Perplexity(5-50) | learning_rate(10-1000)
    # configuring the parameters
    # the number of components = dimension of the embedded space
    # default perplexity = 30 " Perplexity balances the attention t-SNE gives to local and global aspects of the data.
    # It is roughly a guess of the number of close neighbors each point has. ..a denser dataset ... requires higher perplexity value"
    # default learning rate = 200 "If the learning rate is too high, the data may look like a ‘ball’ with any point
    # approximately equidistant from its nearest neighbours. If the learning rate is too low,
    # most points may look compressed in a dense cloud with few outliers."
    tsne_data = model.fit_transform(
        pca_fit)  # When there are more data points, trainX should be the first couple hundred points so TSNE doesn't take too long
    x = tsne_data[:, 0]
    y = tsne_data[:, 1]
    title = "PCA Reduced and T-SNE Plotted with Predicted Latent Points \nPerplexity: " + str(
        perplexity) + "\nLearning Rate: " + str(learning_rate) + "\nLatent Space Dimensionality: " + str(
        latent_dimensionality)
    return x, y, title

########################################################################################################################
# Latent Feature Cluster for Training Data using T-SNE
def TSNE_reduction(latent_points, latent_dimensionality, perplexity=30, learning_rate=20):
    model = TSNE(n_components=2, random_state=0, perplexity=perplexity,
                 learning_rate=learning_rate)  # Perplexity(5-50) | learning_rate(10-1000)
    # configuring the parameters
    # the number of components = dimension of the embedded space
    # default perplexity = 30 " Perplexity balances the attention t-SNE gives to local and global aspects of the data.
    # It is roughly a guess of the number of close neighbors each point has. ..a denser dataset ... requires higher perplexity value"
    # default learning rate = 200 "If the learning rate is too high, the data may look like a ‘ball’ with any point
    # approximately equidistant from its nearest neighbours. If the learning rate is too low,
    # most points may look compressed in a dense cloud with few outliers."
    tsne_data = model.fit_transform(
        latent_points)  # When there are more data points, trainX should be the first couple hundred points so TSNE doesn't take too long
    x = tsne_data[:, 0]
    y = tsne_data[:, 1]
    title = ("T-SNE of Data\nPerplexity: " + str(perplexity) + "\nLearning Rate: "
             + str(learning_rate) + "\nLatent Space Dimensionality: " + str(latent_dimensionality))
    return x, y, title


########################################################################################################################
def plot_dimensionality_reduction(x, y, label_set, title):
    '''
    df = pd.DataFrame({'label': label_set})  # Acts as a grouping variable to produce points with different colors
    color_dict = {'Predicted_Latent_Point': 'red', '': 'blue', '': 'black', '': 'green', '': 'purple'}
    plt.figure(figsize=(8, 6))

    sns.scatterplot(x=x, y=y, hue='label', data=df)
    '''
    plt.title(title)
    if label_set[0].dtype == float:
        plt.scatter(x, y, c=label_set)
        plt.colorbar()
        print("using scatter")
    else:
        for label in set(label_set):
            cond = np.where(np.array(label_set) == str(label))
            plt.plot(x[cond], y[cond], marker='o', linestyle='none', label=label)

        plt.legend(numpoints=1)

    plt.show()
    plt.close()