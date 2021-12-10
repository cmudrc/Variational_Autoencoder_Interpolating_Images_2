import numpy as np
import matplotlib.pyplot as plt
import math
n = 28  # The desired total shape size

density = 1  # Insert the desired number of additional pixel density here, where 0 corresponds to shape thickness of 1
             # and 1 adds a pixel on either side of a shape

A = np.zeros((int(n), int(n)))  # Initializes A matrix with 0 values
A_updated = np.zeros((int(n), int(n)))  # Initializes A matrix with 0 values

print(A[0, 1])
for i in range(n):
    for j in range(n):
        if i == 0 or j == 0 or i == n-1 or j == n-1:
            A[i][j] = 1
        # print(A[i][j])
for i in range(n):
    print("row " + str(i))
    for j in range(n):
        print(A[i][j])

print("Figure 1")
plt.matshow(A, cmap='gray')
plt.title("Original Shape")
plt.colorbar()
plt.show()
Aorig = A

for dens in range(density):
    print("dens" + str(dens))
    for i in range(1, n-1):
        for j in range(1, n-1):
            # print(str(i) + "," + str(j))
            if A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1] > 0:
                A_updated[i][j] = 1
    for i in range(n):
        for j in range(n):
            if A_updated[i][j] == 1:
                A[i][j] = 1  # Replace the A with the new updated A terms, and then perform the density increase again



print("Figure 2")
plt.matshow(A, cmap='gray')
plt.title("Shape with density=" + str(density))
plt.colorbar()
plt.show()

########################################################################################################################
# Used to create figures with a gradient based density change
for dens in range(density-1):
    previous_density = ((density - (dens-1))/(density+1))
    current_density = ((density - dens)/(density+1))
    print("previous dens" + str(previous_density))
    print("current density" + str(current_density))
    for i in range(1, n-1):
        for j in range(1, n-1):
            # print(str(i) + "," + str(j))
            if Aorig[i-1][j] == previous_density or Aorig[i+1][j] == previous_density or Aorig[i][j-1] == previous_density or Aorig[i][j+1] == previous_density:
                A_updated[i][j] = current_density
for k in range(n):
    for z in range(n):
        if A_updated[k][z] != 0:
            A[k][z] = A_updated[k][z]  # Replace the A with the new updated A terms, and then perform the density increase again
for i in range(n):
    print("row " + str(i))
    for j in range(n):
        print(A[i][j])
print("Figure 2")
plt.matshow(A, cmap='gray')
plt.title("Shape with gradient density=" + str(density))
plt.colorbar()
plt.show()
