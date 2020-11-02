from matplotlib import pyplot as plt
import math
import numpy as np

def mish(x):
    return x * math.tanh(math.log(1 + math.exp(x)))


x = np.linspace(-5, 5, 1000)#考虑一下取值，看看怎么取能让图示更美观
y = []
for i in x:
    y.append(mish(i))

#调整xy坐标轴的位置
plt.figure()
ax=plt.gca()
ax.spines['right'].set_color('None')
ax.spines['top'].set_color('None')

ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('axes',0.5))

plt.plot(x, y,label='Mish')
#plt.grid()
plt.legend()
#plt.show()
plt.savefig('mish.svg', format='svg')
