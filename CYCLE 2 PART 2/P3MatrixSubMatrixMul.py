import numpy as np;

x = np.arange(1,31).reshape(5,6)
print("\n x-> big matrix :\n",x)
#x[column,row]
y=x[1:4,2:5]

#print("y-> sub matrix of x : \n",y)

z = np.arange(2,11).reshape(3,3)
#print("matrix z :\n",z)

y = np.multiply(y,z)

#print("multiplied matrix : \n",y)

x[1:4,2:5] = y
print("\n final replaced matrix:\n ",x)
