#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 14:34:19 2019

@author: william
"""
import matplotlib.pyplot as plt
import numpy as np

def y(x, r=1, x0=0):
    return np.sqrt(r**2-(x-x0)**2)

xs = np.linspace(10, 190, 181)
ys = 2*y(xs,90,100)

xs_rm = np.linspace(0, 40, 41)
ys_rm = 2*y(xs_rm, 20, 20)

plt.close('all')

y_plot = np.copy(ys)
#y_plot[70:111]-=ys_rm

fig = plt.figure()
ax = fig.add_subplot(2,1,1)
#ax = fig.add_subplot(1,1,1)
ax.plot(xs, y_plot/np.max(ys), "k")
ax.set_xlim(0, 200)
ax.set_ylim(0, 1.1)
ax.set_xlabel("Pixel")
ax.set_ylabel("Intensity")

#y_plot = np.copy(ys)
#y_plot[20:61]-=ys_rm
#
#ax = fig.add_subplot(2,1,2)
#ax.plot(xs, y_plot/np.max(ys), "k")
#ax.set_xlim(0, 200)
#ax.set_ylim(0, 1.1)
#ax.set_xlabel("Pixel")
#ax.set_ylabel("Intensity")
#
#plt.tight_layout()
plt.show()
