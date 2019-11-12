#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 10:16:01 2019

@author: william
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

class Shape():
    def __init__(self, r=1, n=360, centre=(0,0), sig=None, h_centre=None, h_r=None, h_n=None):
        self._r = r
        self._theta = 2*np.pi/n
        self.centre = (centre[0], centre[1])
        self.theta = np.array([self._theta*i for i in range(n+1)])
        if sig is None:
            sig = .1*r
        self.r = np.array([r+np.random.normal(0,sig) for t in self.theta])
        self.r[-1]=self.r[0]
        self.update_coords()
        self.holes = []
        if h_centre is not None:
            if not h_r:
                h_r = .25*self._r
            if not h_n:
                h_n = round(n*h_r/r)
            self.add_hole(h_r, h_n, h_centre, sig)
        
    def add_hole(self, r, n, centre, sig):
        self.holes.append(Shape(r, n, centre, sig))
    
    def update_coords(self, phi=0):
        self.coords = np.array([(l*np.sin(t+phi),l*np.cos(t+phi)) for t,l in zip(self.theta,self.r)])
        self.coords[:,0] += self.centre[0]
        self.coords[:,1] += self.centre[1]
        
    def plot(self, ax, color="#1f77b4"):
        ax.plot(self.coords[:,0], self.coords[:,1], color)
        for hole in self.holes:
            hole.plot(ax)
#        r = self._r
#        ax.set_xlim(-1.1*r,1.1*r)
#        ax.set_ylim(-1.1*r,1.2*r)
        
    def project(self, n, phi=0, lower=None, upper=None, background=0, noise=None, gauss=None):
        if lower is None:
            lower = np.min(self.coords[:,0])
        if upper is None:
            upper = np.max(self.coords[:,0])
        sin = np.sin(phi)
        cos = np.cos(phi)
        c = self.centre
        t = [[] for _ in range(n)]
        n -= 1
        xs = [i*(upper-lower)/n+(lower-upper)/2 for i in range(n+1)]
        for a,b in zip(self.coords[:-1],self.coords[1:]):
            dif = b-a
            for i,x0 in enumerate(xs):
                t1 = ((c[0]+x0*cos-a[0])*dif[1]-(c[1]+x0*sin-a[1])*dif[0])/(cos*dif[0]+sin*dif[1])
                t2 = (c[1]-a[1]+x0*sin+t1*cos)/(dif[1])
                if 0 <= t2 < 1:
                    t[i].append(t1)
        for x in t:
            x.sort()
        t = np.array([sum(a-b for a,b in zip(i[1::2],i[0::2])) for i in t])
        
        for hole in self.holes:
            t -= hole.project(n+1, phi, lower, upper)[0]
        
        #Experimental Noise
        t += background
        if gauss:
            t = gaussian_filter(t, sigma=gauss)
        if noise:
            t += np.random.normal(0, noise*t)
        return t, np.array(xs)
        
    
if __name__ == '__main__':
    #shape = Shape(1, 120, h_centre=np.random.rand(2)*.8-.4, sig=.05)
    plt.close("all")
    ax = plt.subplot(111)
    shape.plot(ax)
    
    t, xs = shape.project(183, 0.0, lower=-1.2, upper=1.2)#, background=20, noise=0.001, gauss=0.5)
    plt.figure()
    ax = plt.subplot(111)
    ax.plot(xs, t)