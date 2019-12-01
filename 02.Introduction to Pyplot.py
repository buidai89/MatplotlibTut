# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 15:30:54 2019

@author: bdai729
"""

"""
*********
intro to pyplot
*********
reference: https://matplotlib.org/tutorials/introductory/pyplot.html#sphx-glr-
tutorials-introductory-pyplot-py

https://matplotlib.org/gallery/index.html#mplot3d-examples-index

"""

import matplotlib.pyplot as plt
import numpy as np

# <codecell> simple plot
plt.plot([1, 2, 3, 4], [5, 6, 7, 8])
plt.ylabel("some numbers")
plt.xlabel("x axis label")
plt.show()

"""
*** Note::
    the pyplot API is generally less-flexible than the object-oriented API. 
    Most of the function calls you see here can also be called as methods from 
    an Axes object.
    TO ME, it might be the best practice to use an axes (or an axes_list) with 
    their methods. see below
"""
# <codecell> My preference
fig, axes = plt.subplots(1, 1, figsize=(9, 6))
fig.subplots_adjust(bottom=0.15, left=0.2)
axes.plot([1, 2, 3, 4], [1, 2, 3, 4])
axes.set_title("Plot using axes objects and their methods")
axes.set_xlabel("X label")
axes.set_ylabel("Y label")
axes.set_xlim(0, 4)
axes.set_ylim(1, 4)
axes.grid()

fig, ax_lst = plt.subplots(2, 2)
ax_lst[0, 1].plot([1, 2, 3, 4], [1, 2, 3, 4])
ax_lst[0, 1].set_title("subplot 2")
ax_lst[0, 1].set_xlabel("X label")
ax_lst[0, 1].set_ylabel("Y label")
ax_lst[0, 1].set_xlim(0, 4)
ax_lst[0, 1].set_ylim(1, 4)
ax_lst[0, 1].grid()
fig.suptitle("access subtitle of a figure object")

ax_lst[1, 0].scatter([1, 2, 3, 4], [1, 1.5, 3.5, 4])

# <codecell> formating plot style
# reference: https://matplotlib.org/api/_as_gen/matplotlib.pyplot.
# plot.html#matplotlib.pyplot.plot
fig, axes = plt.subplots(1, 1)
axes.plot([1, 2, 3, 4], [1, 2, 3, 4], 'go--', linewidth=2, markersize=12)
axes.set_title("sub plot title")
axes.set_xlabel("X label")
axes.set_ylabel("Y label")
axes.set_xlim(0, 4)
axes.set_ylim(1, 4)
axes.grid()

# <codecell> Plotting with keyword strings
data = {'a': np.arange(50),
        'c': np.random.randint(0, 50, 50),
        'd': np.random.randn(50)}
data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 100

plt.scatter('a', 'b', c='c', s='d', data=data)
plt.xlabel('entry a')
plt.ylabel('entry b')
plt.show()

# <codecell> Plotting with categorical variables
# It is also possible to create a plot using categorical variables. 
# Matplotlib allows you to pass categorical variables directly to many 
# plotting functions. For example:
names = ['group_a', 'group_b', 'group_c']
values = [1, 10, 100]
plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.bar(names, values)
plt.subplot(132)
plt.scatter(names, values)
plt.subplot(133)
plt.plot(names, values, linewidth=5.0)
plt.suptitle('Categorical Plotting')
plt.show()

# <codecell> Working with multiple figures and axes
def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)


t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.figure()
plt.subplot(211)
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

plt.subplot(212)
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
plt.show()

# <codecell> Working with multiple figures and axes
plt.figure(1)                # the first figure
plt.subplot(211)             # the first subplot in the first figure
plt.plot([1, 2, 3])
plt.subplot(212)             # the second subplot in the first figure
plt.plot([4, 5, 6])

plt.figure(2)                # a second figure
plt.plot([4, 5, 6])          # creates a subplot(111) by default

plt.figure(1)                # figure 1 current; subplot(212) still current
plt.subplot(211)             # make subplot(211) in figure1 current
plt.title('Easy as 1, 2, 3')  # subplot 211 title

# <codecell> plot f(x)
# plot y = sin(x)
import numpy as np
fig, axes = plt.subplots(1, 1)
x = np.arange(-np.pi, np.pi, 0.2)
y = np.sin(x)
z = np.cos(x)
axes.plot(x, y)
axes.set_title("y = f(x) = sin(x)")
axes.set_xlabel("X label")
axes.set_ylabel("Y label")
axes.set_xlim(-np.pi, np.pi)
axes.set_ylim(-1, 1)
axes.grid()

plt.figure(1)
plt.subplot(211)
plt.plot(x, y)
plt.subplot(212)
plt.scatter(x, z)
plt.xlim(-5, 5)
plt.ylim(-2, 2)

z2 = 2*np.tan(x)
plt.figure(2)
plt.plot(x, z2)

# <codecell> working with text
import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure()
fig.suptitle('bold figure suptitle', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
ax.set_title('axes title')

ax.set_xlabel('xlabel')
ax.set_ylabel('ylabel')

ax.text(3, 8, 'boxed italics text in data coords', style='italic',
        bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10})

ax.text(2, 6, r'an equation: $E=mc^2$', fontsize=15)

ax.text(3, 2, 'unicode: Institut für Festkörperphysik')

ax.text(0.95, 0.01, 'colored text in axes coords',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=15)


ax.plot([2], [1], 'o')
ax.annotate('annotate', xy=(2, 1), xytext=(3, 4),
            arrowprops=dict(facecolor='black', shrink=0.05))

ax.axis([0, 10, 0, 10])

plt.show()

# <codecell> working with text (cont.)
mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

# the histogram of the data
n, bins, patches = plt.hist(x, 50, density=1, facecolor='g', alpha=0.75)


plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()

# <codecell> font
from matplotlib.font_manager import FontProperties

font = FontProperties()
font.set_family('serif')
font.set_name('Times New Roman')
font.set_style('italic')

fig, ax = plt.subplots(figsize=(5, 3))
fig.subplots_adjust(bottom=0.15, left=0.2)
ax.plot(x1, y1)
ax.set_xlabel('time [s]', fontsize='large', fontweight='bold')
ax.set_ylabel('Damped oscillation [V]', fontproperties=font)

plt.show()

# <codecell> TeX rendering
x1 = np.linspace(0.0, 5.0, 100)
y1 = np.cos(2 * np.pi * x1) * np.exp(-x1)
fig, ax = plt.subplots(figsize=(9, 6))
fig.subplots_adjust(bottom=0.2, left=0.2)
ax.plot(x1, np.cumsum(y1**2))
ax.set_xlabel('time [s] \n This was a long experiment')
ax.set_ylabel(r'$\int\ Y^2\ dt\ \ [V^2 s]$')
plt.show()

# <codecell> anotating text
ax = plt.subplot(111)

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
line, = plt.plot(t, s, lw=2)

plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )
plt.annotate('local min', xy=(2.5, -1), xytext=(3, -1.5),
             arrowprops=dict(facecolor='black', shrink=0.05),
             )

plt.ylim(-2, 2)
plt.show()

# <codecell> logarithmic scale
from matplotlib.ticker import NullFormatter  # useful for `logit` scale

# Fixing random state for reproducibility
np.random.seed(19680801)

# make up some data in the interval ]0, 1[
y = np.random.normal(loc=0.5, scale=0.4, size=1000)
y = y[(y > 0) & (y < 1)]
y.sort()
x = np.arange(len(y))

# plot with various axes scales
plt.figure()

# linear
plt.subplot(221)
plt.plot(x, y)
plt.yscale('linear')
plt.title('linear')
plt.grid(True)


# log
plt.subplot(222)
plt.plot(x, y)
plt.yscale('log')
plt.title('log')
plt.grid(True)


# symmetric log
plt.subplot(223)
plt.plot(x, y - y.mean())
plt.yscale('symlog', linthreshy=0.01)
plt.title('symlog')
plt.grid(True)

# logit
plt.subplot(224)
plt.plot(x, y)
plt.yscale('logit')
plt.title('logit')
plt.grid(True)
# Format the minor tick labels of the y-axis into empty strings with
# `NullFormatter`, to avoid cumbering the axis with too many labels.
plt.gca().yaxis.set_minor_formatter(NullFormatter())
# Adjust the subplot layout, because the logit one may take more space
# than usual, due to y-tick labels like "1 - 10^{-3}"
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

plt.show()

# <codecell> Legend
# reference: https://matplotlib.org/gallery/text_labels_and_annotations/
# figlegend_demo.html#sphx-glr-gallery-text-labels-and-annotations-figlegend-demo-py
import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2)

x = np.arange(0.0, 2.0, 0.02)
y1 = np.sin(2 * np.pi * x)
y2 = np.exp(-x)
l1, = axs[0].plot(x, y1)
l2, = axs[0].plot(x, y2, marker='o')

y3 = np.sin(4 * np.pi * x)
y4 = np.exp(-2 * x)
l3, = axs[1].plot(x, y3, color='tab:green')
l4, = axs[1].plot(x, y4, color='tab:red', marker='^')

fig.legend((l1, l2), ('Line 1', 'Line 2'), 'upper left')
fig.legend((l3, l4), ('Line 3', 'Line 4'), 'upper right')

plt.tight_layout()
plt.show()

# <codecell> Bode plot
from scipy import signal
from lcapy import Circuit, s, j, omega
from matplotlib.pyplot import savefig
from numpy import logspace
from sympy import *
import matplotlib.pyplot as plt

sys = signal.TransferFunction([1600000], [3,  3206400, 2562560000])
w, mag, phase = signal.bode(sys)
plt.figure()
plt.semilogx(w, mag)    # Bode magnitude plot
plt.grid()
plt.figure()
plt.semilogx(w, phase)  # Bode phase plot
plt.grid()
plt.tight_layout()
plt.show()