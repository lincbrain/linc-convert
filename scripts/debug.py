import numpy as np
import psutil
import sparse


def print_mem():
    print(psutil.Process().memory_info().rss/1024/1024)

print_mem()
a = np.broadcast_to(np.zeros(()),(1000,)*3)

print_mem()
b = sparse.random(a.shape,0.0001)
print_mem()
c = a+b
print_mem()