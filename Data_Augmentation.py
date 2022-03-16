import tensorflow as tf
from box_data_set import make_boxes
import numpy as np
from math import pi
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image as im


def Gaussian_Salt_Pepper_Rotation_Shift_Noise():
    print("hello")

# Define the parameters of the Data
image_size = 28  # pixel size of the data you wish you compute (Even numbers required)
number_of_densities = 5  # The number of densities that will be equally spaced between the min and max
min_density = 0  # (Recommend: 0) The minimum density IS NOT included in the data created, it only serves as a placeholder
max_density = 1  # (Recommend: 1) The maximum density IS included in the data created
# Choosing different mins and maxes will require the data to be normalized
box_data = make_boxes(image_size, number_of_densities, min_density, max_density)

def rotate(array, rotation_range):
    rotated_array = tf.keras.preprocessing.image.random_rotation(
        array, rotation_range, row_axis=image_size/2, col_axis=image_size/2, channel_axis=0, fill_mode='nearest',
        cval=0.0, interpolation_order=1
    )
    return rotated_array

def gaussian_noise(array):
    # TensorFlow. 'x' = A placeholder for an image.
    #x = tf.placeholder(dtype=tf.float32, shape=shape)
    array = im.fromarray(array)
    # Adding Gaussian noise
    noise = tf.compat.v1.random.normal(array, mean=0.0, stddev=1.0)
    output = tf.add(array, noise)
    return output


test = box_data[0][0]
print(type(test))
plt.matshow(test, cmap='gray')
plt.title("Before")
plt.colorbar()
plt.show()

test_rotated = gaussian_noise(test)  # np.reshape(test, (2, 2))


'''
def rotate_images(X_imgs, start_angle, end_angle, n_images):
    X_rotate = []
    iterate_at = (end_angle - start_angle) / (n_images - 1)

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3))
    radian = tf.placeholder(tf.float32, shape=(len(X_imgs)))
    tf_img = tf.image.rotate(X, radian)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for index in range(n_images):
            degrees_angle = start_angle + index * iterate_at
            radian_value = degrees_angle * pi / 180  # Convert to radian
            radian_arr = [radian_value] * len(X_imgs)
            rotated_imgs = sess.run(tf_img, feed_dict={X: X_imgs, radian: radian_arr})
            X_rotate.extend(rotated_imgs)

    X_rotate = np.array(X_rotate, dtype=np.float32)
    return X_rotate


# Start rotation at -90 degrees, end at 90 degrees and produce totally 14 images
rotated_imgs = rotate_images(X_imgs, -90, 90, 14)
'''
