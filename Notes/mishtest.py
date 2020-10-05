"""
展示mish函数
date：1005
"""

import math
import numpy as np
from matplotlib import pyplot as plt


def mish(x):
    return x * math.tanh(math.log(1 + math.exp(x)))


def ln_e(x):
    return math.log(1 + math.exp(x))


x = np.linspace(-10, 10, 1000)
y = []
z = []
for i in x:
    y.append(mish(i))
    z.append(ln_e(i))
plt.plot(x, y)
plt.plot(x, z)
plt.grid()
plt.show()
