import numpy as np
import pacmap  # will need to change numba version: pip install numba==0.53
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from smoothness_testing import smoothness
import cv2
from matplotlib.collections import LineCollection


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
def plot_interpolation_smoothness(original_data_latent_points, original_data_labels, interpolated_latent_points,
                                 latent_dimensionality, image_arrays, image_size, number_of_interpolations,
                                  reduction_function=PCA_reduction, markersize=8, marker_color='black',
                                  title="Plot of Latent Points with Interpolated Feature", mesh_predicted_interps=None,
                                  plot_lines=True, plot_points=True, color_bar_min = 85, color_bar_max=100,
                                  plot_row_segments=True, plot_col_segments=True):

    # combines all the latent points of the training data and the interpolation
    train_data_latent_points = np.append(original_data_latent_points, interpolated_latent_points, axis=0)

    # Check if there is an input for mesh_predicted_interps
    if mesh_predicted_interps is not None:
        mesh_predicted_interps = np.reshape(mesh_predicted_interps, (number_of_interpolations, number_of_interpolations,
                                                                     image_size, image_size))
        # reshape so that the images can be indexed by row/column

        # Information needed to plot the smoothness of the rows and columns in the mesh
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

        plt.legend()
        plt.xlabel("Rows/Columns", fontsize=16)
        plt.ylabel("Smoothness (%)", fontsize=16)
        plt.title("Smoothness over mesh ", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ylim([60, 100])
        plt.show()

    # Perform Reduction and get points for Training Images and Predicted Points
    x1, y1, title1 = reduction_function(train_data_latent_points, latent_dimensionality)

    # Get labels for all the points in a single list
    combined_label = original_data_labels  # Contains the labels for all the points
    for i in range(len(interpolated_latent_points)):
        combined_label = np.append(combined_label, np.array("Predicted Points"))

    # Establish plot reduction of images
    image_arrays_padded = np.pad(image_arrays, 1, mode='constant')  # Puts a black box surrounding each array
    fig, ax = plt.subplots()

    # Sort and plot the points and images into the latent space
    for label in set(combined_label):
        cond = np.where(np.array(combined_label) == str(label))
        if label != "Predicted Points":  # Plots the images
            if mesh_predicted_interps is not None:
                image_arrays = np.array(image_arrays)
                image_arrays[image_arrays < 2] = 0.5  # Replaces the training images with gray boxes
                image_arrays_gray = np.pad(image_arrays, 1, mode='constant')  # Puts a black box surrounding each array
                images = image_arrays_gray
            else:
                images = image_arrays_padded
            # Plot the training images
            imscatter(x1[cond], y1[cond], imageData=images[cond], ax=ax, zoom=0.6, image_size=image_size + 2)

        else:
            if plot_points is True:  # Plots the predicted points
                ax.plot(x1[cond], y1[cond], marker='o', c=marker_color, markersize=markersize, linestyle='none',
                        label=label, zorder=5)

    # Pull Coordinates from Reduction for plotting the Mesh
    interpolation_cords_x = x1[-np.shape(interpolated_latent_points)[0]:]  # coordinates of the interpolation points x(ordered)
    interpolation_cords_x = np.reshape(interpolation_cords_x, (np.shape(mesh_predicted_interps)[0], np.shape(mesh_predicted_interps)[1]))

    interpolation_cords_y = y1[-np.shape(interpolated_latent_points)[0]:]  # coordinates of the interpolation points y(ordered)
    interpolation_cords_y = np.reshape(interpolation_cords_y, (np.shape(mesh_predicted_interps)[0], np.shape(mesh_predicted_interps)[1]))

    # Create the Segments between the rows and columns in the Mesh
    row_lines = []
    for row in range(np.shape(interpolation_cords_x)[0]):
        row_lines.append([(interpolation_cords_x[row, 0], interpolation_cords_y[row, 0]),
                          (interpolation_cords_x[row, -1], interpolation_cords_y[row,-1])])
    col_lines = []
    for col in range(np.shape(interpolation_cords_x)[1]):
        col_lines.append([(interpolation_cords_x[0, col], interpolation_cords_y[0, col]),
                          (interpolation_cords_x[-1, col], interpolation_cords_y[-1, col])])

    # Plot the Line Segments in the rows and columns in the mesh
    smoothness_line_row = np.array(smoothness_line_row) / 100
    smoothness_line_col = np.array(smoothness_line_col) / 100

    if plot_row_segments == plot_col_segments == True:  # Plots rows and columns
        plot_line_segments_rows_columns(row_lines, col_lines, smoothness_line_row, smoothness_line_col, ax, "Row", "Column")
        line_segment_title = ": Smoothness of Rows and Columns Represented by Line Segments"

    elif plot_col_segments is True:  # Plots the columns only
        plot_line_segments(col_lines, smoothness_line_col, ax, color_bar_min=color_bar_min,
                           color_bar_max=color_bar_max)  # function that plots the line segments and color codes them
        line_segment_title = ": Smoothness Columns Represented by Line Segments"

    elif plot_row_segments is True:  # Plots the rows only
        plot_line_segments(row_lines, smoothness_line_row, ax, color_bar_min=color_bar_min,
                           color_bar_max=color_bar_max)  # function that plots the line segments and color codes them
        line_segment_title = ": Smoothness of Rows Represented by Line Segments"

    else:  # Recommends the user to use a different function
        print("Use plot_reduction_interpolation if you do not want line segments on your figure")
        line_segment_title = ""

    # Plotting the Images in the 4 Corners of the Mesh
    images_corners = []
    x_corners = []
    y_corners = []
    for point in [(0, 0), (0, -1), (-1, 0), (-1, -1)]:  # Loop through the corner points in the mesh
        print(point)
        images_corners.append(np.pad(mesh_predicted_interps[point], 1, mode='constant')) # Puts a black box surrounding each array
        x_corners.append(interpolation_cords_x[point])
        y_corners.append(interpolation_cords_y[point])
    print(np.shape(images_corners))
    print(np.shape(x_corners))
    imscatter(x_corners, y_corners, imageData=images_corners, ax=ax, zoom=1.5, image_size=image_size + 2)

    # Plots the predicted points, line segments, training images, and images from the 4 corners of the mesh on a
    # single figure
    plt.legend(numpoints=1)
    plt.title(title + line_segment_title)
    plt.show()


########################################################################################################################
def plot_line_segments(segments, smoothness_of_segment, ax, color_bar_min=85, color_bar_max=100):
    # Segments - list of line coordinates
    # Smoothness of Segment - the smoothness of the images over the segment
    # ax - the predefined axis that is being used to plot the data

    # Setup Colorbar Color, Min and Max
    cmap = matplotlib.colormaps['viridis']  # A function that returns the color value of a number (0-1)
    norm = matplotlib.colors.Normalize(vmin=color_bar_min / 100,
                                       vmax=color_bar_max / 100)  # A function to normalize values between a desired min and max

    # Plot the Line segments
    line_segment_rows = LineCollection(segments, colors=cmap(norm(smoothness_of_segment)), linestyles='solid',
                                       zorder=20)
    ax.add_collection(line_segment_rows)
    fig = plt.gcf()

    # Color bar settings for Line Segments
    cbar = fig.colorbar(line_segment_rows,
                        ticks=[0, norm(min(smoothness_of_segment)),norm(max(smoothness_of_segment)), 1])  # Locations of labels on Color Bar
    cbar.set_label('Smoothness (%)')  # Title of the color bar
    cbar.ax.set_yticklabels(
        [str(color_bar_min), str(round(min(smoothness_of_segment) * 100, 2)) + " - Min", str(round(max(smoothness_of_segment) * 100, 2)) + " - Max",  '100'])  # Labels on Color Bar
    ax.autoscale()


########################################################################################################################
def plot_line_segments_rows_columns(segments1, segments2, smoothness_of_segment1, smoothness_of_segment2, ax,
                                    name_segment_1="Segment Set 1", name_segment_2="Segment Set 2",
                                    color_bar_min=85, color_bar_max=100):
    # Used to plot two different sets of segments on the same scale
    # Segments - list of line coordinates
    # Smoothness of Segment - the smoothness of the images over the segment
    # ax - the predefined axis that is being used to plot the data

    # Setup Colorbar Color, Min and Max
    cmap = matplotlib.colormaps['viridis']  # A function that returns the color value of a number (0-1)
    norm = matplotlib.colors.Normalize(vmin=color_bar_min / 100,
                                       vmax=color_bar_max / 100)  # A function to normalize values between a desired min and max
    # Combine Segments for Plotting
    collective_segments = np.append(segments1, segments2,axis=0)
    collective_smoothness = np.append(smoothness_of_segment1,smoothness_of_segment2)

    # Plot the Line segments
    line_segment_rows = LineCollection(collective_segments, colors=cmap(norm(collective_smoothness)), linestyles='solid',
                                       zorder=20, linewidths=2)
    ax.add_collection(line_segment_rows)
    fig = plt.gcf()

    # Color bar settings for Line Segments
    cbar = fig.colorbar(line_segment_rows,
                        ticks=[0,
                               norm(min(smoothness_of_segment1)),
                               norm(max(smoothness_of_segment1)),
                               norm(min(smoothness_of_segment2)),
                               norm(max(smoothness_of_segment2)),
                               1])  # Locations of labels on Color Bar
    cbar.set_label('Smoothness (%)')
    cbar.ax.set_yticklabels(
        [str(color_bar_min),
         str(round(min(smoothness_of_segment1) * 100, 2)) + " - Min of " + name_segment_1,
         str(round(max(smoothness_of_segment1) * 100, 2)) + " - Max of " + name_segment_1,
         str(round(min(smoothness_of_segment2) * 100, 2)) + " - Min of " + name_segment_2,
         str(round(max(smoothness_of_segment2) * 100, 2)) + " - Max of " + name_segment_2,
         '100'])  # Labels on Color Bar
    ax.autoscale()
