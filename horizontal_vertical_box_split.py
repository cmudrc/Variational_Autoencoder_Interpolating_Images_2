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
# for i in range(n):
#     print("row " + str(i))
#     for j in range(n):
#         print(A[i][j])

for dens in range(density):
    for i in range(n):
        for j in range(n):
            if i == math.floor((n-1)/2) or i == math.ceil((n-1)/2):
                A[i][j] = 1
            if j == math.floor((n-1)/2) or j == math.ceil((n-1)/2):
                A[i][j] = 1
print("Figure 1")
plt.matshow(A, cmap='gray')
plt.title("Original Shape")
plt.colorbar()
plt.show()


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

