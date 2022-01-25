import numpy as np
import matplotlib.pyplot as plt
import math
from Piecewise_Box_Functions import back_slash_array, basic_box_array, forward_slash_array, hot_dog_array, hamburger_array, center_array, update_array, add_pixels


def basic_box(additional_pixels, density, image_size):
    A = basic_box_array(image_size)  # Creates the outside edges of the box
    # Increase the thickness of each part of the box
    A = add_pixels(A, additional_pixels, image_size)
    return A*density


def horizontal_vertical_box_split(additional_pixels, density, image_size):
    A = basic_box_array(image_size)  # Creates the outside edges of the box
    # Place pixels across the horizontal and vertical axes to split the box
    A = update_array(A, hot_dog_array(image_size), image_size)
    A = update_array(A, hamburger_array(image_size), image_size)
    # Increase the thickness of each part of the box
    A = add_pixels(A, additional_pixels, image_size)
    return A*density


def diagonal_box_split(additional_pixels, density, image_size):
    A = basic_box_array(image_size)  # Creates the outside edges of the box

    # Add pixels along the diagonals of the box
    A = update_array(A, back_slash_array(image_size), image_size)
    A = update_array(A, forward_slash_array(image_size), image_size)

    # Adds pixels to the thickness of each component of the box
    # Increase the thickness of each part of the box
    A = add_pixels(A, additional_pixels, image_size)
    return A*density


n = 28  # The desired total shape size
desired_number_of_additional_pixels = 4
desired_density = 1 # Will be used to create matrices with various densities, ranges from 0 to 100, representing null space and fully solid space respectively
B = diagonal_box_split(desired_number_of_additional_pixels, desired_density, n)
print("Figure")
plt.matshow(B, cmap='gray')
plt.title("Shape with " + str(desired_number_of_additional_pixels) + " Additonal Pixel(s)")
plt.colorbar()
plt.show()

