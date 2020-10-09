"""
date： 1009 学习一下画图



"""
from matplotlib import pyplot as plt
import math
import numpy as np

def mish(x):
    return x * math.tanh(math.log(1 + math.exp(x)))


x = np.linspace(-10, 10, 1000)
y = []
for i in x:
    y.append(mish(i))

ax=plt.gca()
ax.spines['right'].set_color('None')
ax.spines['top'].set_color('None')

ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('axes',0.5))

plt.plot(x, y)
#plt.grid()
plt.show()


ax.set_aspect('equal') 
ax.minorticks_on() 
ax.set_xlim(0,16) 
ax.grid(which='minor', axis='both')

ax.xaxis.set_tick_params(rotation=45,labelsize=18,colors='w') 
start, end = ax.get_xlim() 
ax.xaxis.set_ticks(np.arange(start, end,1)) 
ax.yaxis.tick_right()
