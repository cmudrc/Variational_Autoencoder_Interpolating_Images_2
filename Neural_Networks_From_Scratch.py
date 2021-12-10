import numpy as np
import matplotlib.pyplot as plt
inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0, 2.0],
          [-1.5, 2.7, 3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

output = np.dot(inputs, np.array(weights).T) + biases
print(output)
"""
layer_outputs = []  # output of current layer
for neuron_weights, neuron_bias in zip(weights, biases):  # this combines weights and biases
    # into multiple lists element-wise
    # names these lists neuron_weights and neuron_bias for collective
    neuron_output = 0  # output of a given neuron
    for n_input, weight in zip(inputs, neuron_weights):  # this will zip weights, inputs and biases together
        neuron_output += n_input*weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

print(layer_outputs)
"""
