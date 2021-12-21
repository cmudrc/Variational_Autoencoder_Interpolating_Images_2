import numpy as np
import matplotlib.pyplot as plt

from basic_box import basic_box


image_size = 28  # number of pixels on 2D plane
maximum_additional_pixels = (n/2)-1  # this will calculate the amount that the box will need to become completely filled

step_number = 5
min_density = 0
max_density = 1
density_increment = (max_density-min_density)/step_number


for i in range(1, step_number+1):
    density = i*density_increment  # causing some sort of decimal float issue
    # Will be used to create matrices with various densities, ranges from 0 to 1, representing null space and fully solid space respectively
    print(density)
    for j in range(maximum_additional_pixels):
        number_of_additional_pixels = j

# and 1 adds a pixel on either side of a shape
density = 1
A = basic_box(maximum_additional_pixels, density, image_size)
print("Figure" + str(density))
plt.matshow(A, cmap='gray')
plt.title("Shape with " + str(number_of_additional_pixels) + " Additonal Pixel(s)")
plt.colorbar()
plt.show()