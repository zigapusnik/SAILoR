import numpy as np

a = np.matrix([[0, 1, 1, 1, 0],[0, 0, 1, 0, 1],[0, 0, 0, 0, 1],[0, 0, 1, 0, 1], [0, 0, 0, 0, 0]])

b = np.matmul(a, a)

c = np.matmul(a, b)

print(a)
print(b)
print(c)