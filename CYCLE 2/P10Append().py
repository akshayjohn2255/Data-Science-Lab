import numpy as np

ar = np.array([1, 2, 3, 4, 5, 6, 7])
a = np.append(ar,6)
print("1D array",ar)
print("the use of append() function in 1D array \n",a)

 
ar1 = np.array( [ [1, 2, 3],[ 4, 5, 6] ])
print("2D array : \n", ar1)

a = np.append(ar1,np.array([[7,8,9]]),axis=0)
print("\n2D array")
print(ar1)
print("the use of append() function in 2D array")
print(a)
