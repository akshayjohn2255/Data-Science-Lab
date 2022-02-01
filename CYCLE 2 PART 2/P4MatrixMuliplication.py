import numpy as np
  
A = ([1, 1, 1],[1 ,1, 1],[1, 1, 1])
B = ([1, 1, 1],[1 ,1, 1],[1, 1, 1])
C = ([1, 1, 1],[1 ,1, 1],[1, 1, 1])

res = np.dot(A,np.dot(B,C))

print("\n Product Of Matrices: \n\n", res)
