import numpy as np

x = np.array([[2, 4, 6], [6, 8, 10]], dtype = complex )

print("original Array")

print(x)

print("No.of rows and columns=",x.shape)

print( "Dimension=" ,x.ndim)

newarr = x.reshape(3,2)

print(" Reshaped Array ")

print(newarr)
