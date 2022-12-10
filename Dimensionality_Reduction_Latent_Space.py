import numpy as np
import pacmap  # will need to change numba version: pip install numba==0.53
# import seaborn as sns
# import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import matplotlib
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from smoothness_testing import smoothness
import cv2
from matplotlib.collections import LineCollection
from matplotlib import cm

########################################################################################################################
# Latent Feature Cluster for Training Data using PaCMAP
def PaCMAP_reduction(latent_points, latent_dimensionality, random_state=1):
    # initializing the pacmap instance
    X = latent_points

    embedding = pacmap.PaCMAP(n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, random_state=random_state)

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
    pca = PCA(n_components=2, random_state=0)
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


########################################################################################################################
# Scatter with images instead of points
def imscatter(x, y, ax, imageData, image_size, zoom):
    images = []
    for i in range(len(x)):
        x0, y0 = x[i], y[i]
        # Convert to image
        img = imageData[i] * 255.
        img = img.astype(np.uint8).reshape([image_size, image_size])
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # Note: OpenCV uses BGR and plt uses RGB
        image = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
        images.append(ax.add_artist(ab))

    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()


# Plot images in latent space with respective reduction method
def Latent_Image_Proj(image_arrays, image_size,train_latent_points, latent_dimensionality, reduction_function=PCA_reduction):
    # Compute Reduction embedding of latent space
    x, y, title = reduction_function(train_latent_points, latent_dimensionality)
    # Plot images according to reduction embedding
    image_arrays = np.pad(image_arrays, 1, mode='constant')
    fig, ax = plt.subplots()
    imscatter(x, y, imageData=image_arrays, ax=ax, zoom=0.6, image_size=image_size+2)
    plt.title(title)
    plt.show()


########################################################################################################################
def plot_reduction_interpolation(original_data_latent_points, original_data_labels, interpolated_latent_points,
                                 latent_dimensionality, image_arrays, image_size, reduction_function=PCA_reduction, markersize=8, marker_color='red',
                                 title="Plot of Latent Points with Interpolated Feature", plot_lines=True, plot_points=True):
    train_data_latent_points = np.append(original_data_latent_points, interpolated_latent_points, axis=0)
    print("Shape of combined points", np.shape(train_data_latent_points))

    x1, y1, title1 = reduction_function(train_data_latent_points, latent_dimensionality)

    combined_label = original_data_labels
    for i in range(len(interpolated_latent_points)):
        combined_label = np.append(combined_label, np.array("Predicted Points"))

    # Establish plot reduction of images
    image_arrays = np.pad(image_arrays, 1, mode='constant')
    fig, ax = plt.subplots()

    # Sort and plot the points and images into the latent space
    for label in set(combined_label):
        cond = np.where(np.array(combined_label) == str(label))
        if label != "Predicted Points":
            imscatter(x1[cond], y1[cond], imageData=image_arrays[cond], ax=ax, zoom=0.6, image_size=image_size + 2)

        else:
            if plot_points is True:
                ax.plot(x1[cond], y1[cond], marker='o', c=marker_color, markersize=markersize, linestyle='none',
                        label=label, zorder=10)
            if plot_lines:
                ax.plot(x1[cond], y1[cond], 'ro-', zorder=10)

    plt.legend(numpoints=1)
    plt.title(title)
    plt.show()


########################################################################################################################
def plot_interpolation_smoothness(original_data_latent_points, original_data_labels, interpolated_latent_points, mesh_predicted_interps,
                                 latent_dimensionality, image_arrays, image_size, number_of_interpolations, reduction_function=PCA_reduction, markersize=8, marker_color='red',
                                 title="Plot of Latent Points with Interpolated Feature", plot_lines=True, plot_points=True, interp_type="mesh"):

    train_data_latent_points = np.append(original_data_latent_points, interpolated_latent_points, axis=0)
    print("Shape of combined points", np.shape(train_data_latent_points))

    if interp_type == "mesh":
        mesh_predicted_interps = np.reshape(mesh_predicted_interps, (
        number_of_interpolations, number_of_interpolations, image_size, image_size)) # reshape so that the images can be indexed by row/column



        # Get the smoothness of each row in the mesh
        count_row = []
        smoothness_line_row = []
        for row in range(np.shape(mesh_predicted_interps)[0]):
            count_row.append(row)
            interpolation = mesh_predicted_interps[row, :]
            smoothness_line_row.append(smoothness(interpolation)[0]) # adds the average smoothness to our array
        plt.scatter(count_row, smoothness_line_row, label="Row Smoothness")

        # Get the smoothness for each column in the mesh
        count_col = []
        smoothness_line_col = []
        for col in range(np.shape(mesh_predicted_interps)[1]):
            count_col.append(col)
            interpolation = mesh_predicted_interps[:, col]
            smoothness_line_col.append(smoothness(interpolation)[0])  # adds the average smoothness to our array

        plt.scatter(count_col, smoothness_line_col, label="Column Smoothness")
        plt.xlabel("Rows/Columns", fontsize=16)
        plt.ylabel("Smoothness (%)", fontsize=16)
        plt.title("Smoothness over mesh ", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim([60, 100])
        plt.show()
    #add column
    # interpolated_latent_points[:, col]

    x1, y1, title1 = reduction_function(train_data_latent_points, latent_dimensionality)

    combined_label = original_data_labels
    for i in range(len(interpolated_latent_points)):
        combined_label = np.append(combined_label, np.array("Predicted Points"))

    # Establish plot reduction of images
    image_arrays = np.pad(image_arrays, 1, mode='constant')
    fig, ax = plt.subplots()
    ax1 = ax
    # Sort and plot the points and images into the latent space
    for label in set(combined_label):
        cond = np.where(np.array(combined_label) == str(label))
        if label != "Predicted Points":
            imscatter(x1[cond], y1[cond], imageData=image_arrays[cond], ax=ax, zoom=0.6, image_size=image_size + 2)

        else:
            # print("x , y")
            if plot_points is True:
                ax.plot(x1[cond], y1[cond], marker='o', c=marker_color, markersize=markersize, linestyle='none',
                        label=label, zorder=5)
                # print(x1[cond], y1[cond])
            if plot_lines:
                ax.plot(x1[cond], y1[cond], 'ro-', zorder=10)

    interpolation_cords_x = x1[-np.shape(interpolated_latent_points)[0]:]  # coordinates of the interpolation points (ordered)
    interpolation_cords_x = np.reshape(interpolation_cords_x, (np.shape(mesh_predicted_interps)[0], np.shape(mesh_predicted_interps)[1]))

    interpolation_cords_y = y1[-np.shape(interpolated_latent_points)[0]:]  # coordinates of the interpolation points (ordered)
    interpolation_cords_y = np.reshape(interpolation_cords_y, (np.shape(mesh_predicted_interps)[0], np.shape(mesh_predicted_interps)[1]))

    # colors = pl.cm.jet(np.linspace(0, 1, 20)) # np.shape(mesh_predicted_interps)[0]+np.shape(mesh_predicted_interps)[1]))
    # print(np.shape(colors))
    # print(colors)
    # print(type(colors))
    # print(interpolation_cords_y)
    # print("x, y")
    row_lines = []
    for row in range(np.shape(interpolation_cords_x)[0]):
        row_lines.append([(interpolation_cords_x[row,0], interpolation_cords_y[row,0]), (interpolation_cords_x[row,-1], interpolation_cords_y[row,-1])])

    # viridis = cm.get_cmap('viridis', 12)
    viridis = matplotlib.colormaps['viridis']
    smoothness_line_row = np.array(smoothness_line_row) / 100
    # norm = matplotlib.colors.Normalize(vmin=min(smoothness_line_row), vmax=1)
    norm = matplotlib.colors.Normalize(vmin=.8, vmax=1)
    print("First smooothness value",smoothness_line_row[0])
    print("Norm of first smoothness value",norm(smoothness_line_row[0]))
    print(smoothness_line_row)

    line_segment_rows = LineCollection(row_lines, colors=viridis(norm(smoothness_line_row)), linestyles='solid', zorder=20)
    # line_segment_rows = LineCollection(row_lines, colors=viridis(smoothness_line_row), linestyles='solid', zorder=20)

    ax.add_collection(line_segment_rows)
    fig = plt.gcf()
    cbar = fig.colorbar(line_segment_rows, ticks = [0, viridis(norm(min(smoothness_line_row))),1])
    cbar.set_label('Smoothness (%)')
    # cbar.ax.set_yticklabels([str(round((min(smoothness_line_row)*100),2)), '100'])  # vertically oriented colorbar
    cbar.ax.set_yticklabels(['80',str(round(min(smoothness_line_row)*100,2)) ,'100'])  # vertically oriented colorbar
    ax.set_title('Line Collection with mapped colors')
    # plt.clim(vmin=min(smoothness_line_row), vmax=100)
    ax.autoscale()
        # print("x", interpolation_cords_x[row,0], interpolation_cords_x[row,-1])
        # print("y", interpolation_cords_y[row,0],interpolation_cords_y[row,-1])
        # ax1.plot([interpolation_cords_x[row,0], interpolation_cords_x[row,-1]], [interpolation_cords_y[row,0],interpolation_cords_y[row,-1]], color=np.where(colors == smoothness_line_row[row]/100), zorder=10, label="Smoothness of Rows")
        # plt.colorbar()



    # for col in range(np.shape(interpolation_cords_x)[1]):

    plt.legend(numpoints=1)
    plt.title(title)
    plt.show()
