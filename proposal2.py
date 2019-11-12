#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:57:21 2019

@author: william
"""
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

data = np.array([[1,1,0,1],
                 [1,1,1,1],
                 [1,1,1,0],
                 [1,1,1,1]])

# create discrete colormap
cmap = colors.ListedColormap(['white', 'grey'])
bounds = [0,0.5,1]
norm = colors.BoundaryNorm(bounds, cmap.N)

plt.close("all")
fig, axs = plt.subplots(2,2,gridspec_kw={'width_ratios': [3, 1], 'height_ratios':[3,1]})

ax=axs[-1,-1]
ax.axis('off')

ax=axs[0,0]
ax.imshow(data, cmap=cmap, norm=norm)
ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
ax.set_xticks(np.arange(-.5, len(data[0]), 1));
ax.set_yticks(np.arange(-.5, len(data), 1));
ax.set_yticklabels([])
ax.set_xticklabels([])

ax=axs[0,1]
xs = np.array([sum(r) for r in data])
ys = np.array([c for c in range(len(data))])
ax.barh(ys[::-1]+.5, xs, 1, color="grey")
ax.grid(which='major', axis='y', linestyle='-', color='k', linewidth=1)
ax.set_xticks(np.arange(0, len(data)+1, 1))
ax.set_xlim(0, len(data)+1)
ax.set_yticks(np.arange(0, len(data[0]), 1))
ax.set_ylim(0, len(data))
ax.set_xlabel("Intensity")
ax.set_ylabel("Pixel")

ax=axs[1,0]
xs = np.array([sum([r[c] for r in data]) for c in range(len(data[0]))])
ys = np.array([c for c in range(len(data[0]))])
ax.bar(ys+.5, xs, 1, color="grey")
ax.grid(which='major', axis='x', linestyle='-', color='k', linewidth=1)
ax.set_xticks(np.arange(0, len(data[0]), 1))
ax.set_xlim(0, len(data[0]))
ax.set_yticks(np.arange(0, len(data)+1, 1))
ax.set_ylim(0, len(data)+1)
ax.set_xlabel("Pixel")
ax.set_ylabel("Intensity")

fig.set_figheight(6)
fig.set_figwidth(6)
plt.show()
