import numpy as np
x=np.arange(1,10).reshape(3,3)
print("Matrix:\n ",x)

print("Cube using multiply():\n",np.multiply(x,3))

print("Cube using * :\n ",x*x*x)

print("Cube using power():\n",np.power(x,3))

print("Cube using ** :\n",x**3)

print("Identity matrix of the given square matrix:")
print(np.identity(3,dtype= int))

print(" Display each element of the matrix to different powers.")
print(np.power(x,x))

print("Create a matrix Y with same dimension as X and perform the operation X2+2Y:",)
y = np.arange(11,20).reshape(3,3)
print(np.add((np.power(x,2)),(np.multiply(y,2))))