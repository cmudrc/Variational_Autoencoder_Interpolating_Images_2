import numpy as np
import matplotlib.pyplot as plt
import math
from Piecewise_Box_Functions import back_slash_array, basic_box_array, forward_slash_array, hot_dog_array, hamburger_array, center_array, update_array, add_pixels


def back_slash_box(additional_pixels, density, image_size):
    A = basic_box_array(image_size)  # Initializes A matrix with 0 values
    A = update_array(A, back_slash_array(image_size), image_size)
    A = add_pixels(A, additional_pixels, image_size)
    return A * density


def forward_slash_box(additional_pixels, density, image_size):
    A = basic_box_array(image_size)  # Initializes A matrix with 0 values
    A = update_array(A, forward_slash_array(image_size), image_size)
    A = add_pixels(A, additional_pixels, image_size)
    return A * density


def hot_dog_box(additional_pixels, density, image_size):
    A = basic_box_array(image_size)  # Initializes A matrix with 0 values
    A = update_array(A, hot_dog_array(image_size), image_size)
    A = add_pixels(A, additional_pixels, image_size)
    return A * density


def hamburger_box(additional_pixels, density, image_size):
    A = basic_box_array(image_size)  # Initializes A matrix with 0 values
    A = update_array(A, hamburger_array(image_size), image_size)
    A = add_pixels(A, additional_pixels, image_size)
    return A * density


def x_plus_box(additional_pixels, density, image_size):
    A = basic_box_array(image_size)  # Initializes A matrix with 0 values
    A = update_array(A, hot_dog_array(image_size), image_size)
    A = update_array(A, hamburger_array(image_size), image_size)
    A = update_array(A, forward_slash_array(image_size), image_size)
    A = update_array(A, back_slash_array(image_size), image_size)
    A = add_pixels(A, additional_pixels, image_size)
    return A * density


def forward_slash_plus_box(additional_pixels, density, image_size):
    A = basic_box_array(image_size)  # Initializes A matrix with 0 values
    A = update_array(A, hot_dog_array(image_size), image_size)
    A = update_array(A, hamburger_array(image_size), image_size)
    A = update_array(A, forward_slash_array(image_size), image_size)
    # A = update_array(A, back_slash_array(image_size), image_size)
    A = add_pixels(A, additional_pixels, image_size)
    return A * density


def back_slash_plus_box(additional_pixels, density, image_size):
    A = basic_box_array(image_size)  # Initializes A matrix with 0 values
    A = update_array(A, hot_dog_array(image_size), image_size)
    A = update_array(A, hamburger_array(image_size), image_size)
    # A = update_array(A, forward_slash_array(image_size), image_size)
    A = update_array(A, back_slash_array(image_size), image_size)
    A = add_pixels(A, additional_pixels, image_size)
    return A * density


def x_hot_dog_box(additional_pixels, density, image_size):
    A = basic_box_array(image_size)  # Initializes A matrix with 0 values
    A = update_array(A, hot_dog_array(image_size), image_size)
    # A = update_array(A, hamburger_array(image_size), image_size)
    A = update_array(A, forward_slash_array(image_size), image_size)
    A = update_array(A, back_slash_array(image_size), image_size)
    A = add_pixels(A, additional_pixels, image_size)
    return A * density


def x_hamburger_box(additional_pixels, density, image_size):
    A = basic_box_array(image_size)  # Initializes A matrix with 0 values
    # A = update_array(A, hot_dog_array(image_size), image_size)
    A = update_array(A, hamburger_array(image_size), image_size)
    A = update_array(A, forward_slash_array(image_size), image_size)
    A = update_array(A, back_slash_array(image_size), image_size)
    A = add_pixels(A, additional_pixels, image_size)
    return A * density


def center_box(additional_pixels, density, image_size):
    A = basic_box_array(image_size)  # Initializes A matrix with 0 values
    A = update_array(A, center_array(image_size), image_size)
    A = add_pixels(A, additional_pixels, image_size)
    return A * density
# n = 28  # The desired total shape size
# desired_number_of_additional_pixels = 5
# desired_density = 100 # Will be used to create matrices with various densities, ranges from 0 to 100, representing null space and fully solid space respectively
# B = horizontal_vertical_box_split(desired_number_of_additional_pixels, desired_density, n)
# print("Figure")
# plt.matshow(B, cmap='gray')
# plt.title("Shape with " + str(desired_number_of_additional_pixels) + " Additonal Pixel(s)")
# plt.colorbar()
# plt.show()


'''
# The Plot of each of the boxes
image_size = 28  # The desired total shape size
desired_additional_pixels = 0 # Will add pixels next to each square in the box
desired_density = 1  # Determines the grayscale value

plot_rows = 2
plot_columns = 5
plt.subplot(plot_rows, plot_columns, 1), plt.imshow(back_slash_box(desired_additional_pixels, desired_density, image_size), cmap='gray')
plt.title("back_slash_box")

plt.subplot(plot_rows, plot_columns, 2), plt.imshow(forward_slash_box(desired_additional_pixels, desired_density, image_size), cmap='gray')
plt.title("forward_slash_box")

plt.subplot(plot_rows, plot_columns, 3), plt.imshow(hot_dog_box(desired_additional_pixels, desired_density, image_size), cmap='gray')
plt.title("hot_dog_box")

plt.subplot(plot_rows, plot_columns, 4), plt.imshow(hamburger_box(desired_additional_pixels, desired_density, image_size), cmap='gray')
plt.title("hamburger_box")

plt.subplot(plot_rows, plot_columns, 5), plt.imshow(x_plus_box(desired_additional_pixels, desired_density, image_size), cmap='gray')
plt.title("x_plus_box")

plt.subplot(plot_rows, plot_columns, 6), plt.imshow(forward_slash_plus_box(desired_additional_pixels, desired_density, image_size), cmap='gray')
plt.title("forward_slash_plus_box")

plt.subplot(plot_rows, plot_columns, 7), plt.imshow(back_slash_plus_box(desired_additional_pixels, desired_density, image_size), cmap='gray')
plt.title("back_slash_plus_box")

plt.subplot(plot_rows, plot_columns, 8), plt.imshow(x_hot_dog_box(desired_additional_pixels, desired_density, image_size), cmap='gray')
plt.title("x_hot_dog_box")

plt.subplot(plot_rows, plot_columns, 9), plt.imshow(x_hamburger_box(desired_additional_pixels, desired_density, image_size), cmap='gray')
plt.title("x_hamburger_box")

plt.subplot(plot_rows, plot_columns, 10), plt.imshow(center_box(desired_additional_pixels, desired_density, image_size), cmap='gray')
plt.title("center_box")

plt.show()
'''