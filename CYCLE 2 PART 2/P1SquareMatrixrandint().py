import numpy as np
print("Matrix:")
ar = np.random.randint(10, size=(3, 3))
print(ar)

print("\n Inverse Matrix:")
print(np.linalg.inv(ar))

print("\n Rank of Matrix:")
print(np.linalg.matrix_rank(ar))

print("\n Determinant:")
print(np.linalg.det(ar))

print("\n Transform matrix into 1D array:")
print(np.ravel(ar))

print("\n Eigen values and vectors:")
print(np.linalg.eig(ar))