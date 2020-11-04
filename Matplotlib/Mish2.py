"""
1104 给图画上坐标轴箭头

"""

from matplotlib import pyplot as plt
import math
import numpy as np
import mpl_toolkits.axisartist as axisartist

def mish(x):
    return x * math.tanh(math.log(1 + math.exp(x)))


x = np.linspace(-5, 5, 1000)#考虑一下取值，看看怎么取能让图示更美观
y = []
for i in x:
    y.append(mish(i))


fig = plt.figure()

#figure(num, figsize, dpi, facecolor, edgecolor, frameon, FigureClass, clear)
#新建了一个画板（画图视窗）
ax = axisartist.Subplot(fig, 1,1,1)
fig.add_axes(ax)
#新建一个轴系图（绘图区）对象ax,并添加到画板中

ax.axis[:].set_visible(False)

ax.axis["x"] = ax.new_floating_axis(0, 0)
ax.axis["y"] = ax.new_floating_axis(1, 0)
# new_floating_axis(self, nth_coord, value, axis_direction='bottom')
# 新建可移动的坐标轴
ax.axis["x"].set_axis_direction('top')
ax.axis["y"].set_axis_direction('left')
# ax.axis['x'].label.set_text('x')
# ax.axis['y'].label.set_text('y')

ax.axis["x"].set_axisline_style("->", size=2.0)
ax.axis["y"].set_axisline_style("->", size=2.0)
# ax.axis["x"].set_axis_direction('top')
# ax.axis["y"].set_axis_direction('left')

ax.set_xticks([-4,-2,2,4])
ax.set_yticks([1,2,3,4])

plt.plot(x, y,label='Mish')

plt.legend()
plt.savefig('mish.svg', format='svg')
plt.show()
