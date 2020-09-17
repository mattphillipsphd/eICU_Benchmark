"""
https://towardsdatascience.com/animations-with-matplotlib-d96375c5442c
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')


#print("Creating sine wave animation...")
#fig = plt.figure()
#ax = plt.axes(xlim=(0, 4), ylim=(-2, 2))
#line, = ax.plot([], [], lw=3)
#
#def init():
#    line.set_data([], [])
#    return line,
#
#def animate(i):
#    x = np.linspace(0, 4, 1000)
#    y = np.sin(2 * np.pi * (x - 0.01 * i))
#    line.set_data(x, y)
#    return line,
#
#anim = FuncAnimation(fig, animate, init_func=init,
#                               frames=200, interval=20, blit=True)
#
#anim.save('/home/matt/temp/sine_wave.gif', writer='imagemagick')
#plt.close()
#print("...Done")
#
########################################
# See
# https://stackoverflow.com/questions/9401658/how-to-animate-a-scatter-plot

print("Creating scatter plot animation...")
fig = plt.figure(figsize=(16,16))
ax = plt.axes()
scat = ax.scatter([], [], s=16)
n_frames = 200

def init_scatter():
    scat.set_offsets([], [])
    return scat

def animate_scatter(i):
    P = np.random.rand(100,2) * (1 - i/(2*n_frames))
    scat.set_offsets(P)
    return scat

anim = FuncAnimation(fig, animate_scatter, init_func=init_scatter,
                               frames=n_frames, interval=20, blit=True)

anim.save('/home/matt/temp/scatter.gif', writer='imagemagick')
plt.close()
print("...Done")

