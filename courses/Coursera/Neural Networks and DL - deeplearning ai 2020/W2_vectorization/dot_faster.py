import time

import numpy as np

a = np.random.rand(100000)
b = np.random.rand(100000)

tic = time.time()
c = np.dot(a, b)
toc = time.time()

print(c)
print("Vectorized version: {} ms".format(1000*(toc-tic)))

c = 0
tic = time.time()
for i in range(100000):
    c += a[i]*b[i]
toc = time.time()

print(c)
print("Non-vectorized version: {} ms".format(1000*(toc-tic)))

# 25114.9903654
# Vectorized version: 0.1881122589111328 ms
# 25114.9903654
# Non-vectorized version: 62.642812728881836 ms
