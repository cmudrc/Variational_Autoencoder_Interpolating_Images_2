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

image_size = 28  # The desired total shape size
# and 1 adds a pixel on either side of a shape
desired_additional_pixels = 7
desired_density = 0.2 # Will be used to create matrices with various densities, ranges from 0 to 100, representing null space and fully solid space respectively
B = center_box(desired_additional_pixels, desired_density, image_size)
print("Figure")
plt.matshow(B, cmap='gray')
plt.colorbar()
plt.show()
