from numba import jit
import random
import math
import time
import numpy as np


@jit
def some_func(n):
    z = 0
    for i in range(n):
        x = random.random()
        y = random.random()

        z += (x**2 + y**2) ** 0.5


t0 = time.time()
some_func(1_000_000)
t1 = time.time()

print(t1 - t0)
