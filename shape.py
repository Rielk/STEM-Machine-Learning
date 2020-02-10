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
    def __init__(self, r=1, n=360, centre=(0,0), sig=None,
                 h_centre=None, h_r=None, h_n=None, density=1,
                 s_r=None, s_centre=None, s_n=None, s_density=1,
                 s_s=.04, s_t=(.01, .1), s_clock=True, s_theta=0.):
        self.density = density
        self._r = r
        self.centre = np.array([centre[0], centre[1]])
        if sig is None:
            sig = .1*r
        if r >=0 :
            self._theta = 2*np.pi/n
            self.theta = np.array([self._theta*i for i in range(n+1)])
            self.r = np.array([r+np.random.normal(0,sig) for t in self.theta])
            self.r[-1]=self.r[0]
        
        self.holes = []
        if h_centre is not None:
            if not h_r:
                h_r = .1*self._r
            if not h_n:
                h_n = round(n*h_r/r)
            h_centre = (h_centre[0]+centre[0], h_centre[1]+centre[1])
            self.add_hole(h_r, h_n, h_centre, sig)
        
        self.spirals = []
        if s_centre is not None:
            if not s_r:
                s_r = .8*self._r
            if not s_n:
                s_n = n
            s_centre = (s_centre[0]+centre[0], s_centre[1]+centre[1])
            self.add_spiral(s_r, s_s, s_t, s_clock, s_n, s_centre, s_density, s_theta)
        
        self.rotate_coords(0, about=(.0,.0))
        
    def add_hole(self, r, n, centre, sig):
        self.holes.append(Shape(r, n, centre, sig, density=-self.density))
        dr = np.sqrt(r**2+self._r**2)-self._r
        self._r += dr
        self.r += dr
    
    def add_spiral(self, r, s, t, clock, n, centre, density, dtheta):
        self.spirals.append(Spiral(r, s, t, clock, n, dtheta, centre, density))
    
    def rotate_coords(self, phi=0.0, about=None):
        """
        If about is None, rotate about the centre of the shape,
        If about is a coordinate, rotate about there
        """
        self.phi = phi
        self.coords = np.array([(l*np.sin(t+phi),l*np.cos(t+phi)) for t,l in zip(self.theta,self.r)])
        if about is None:
            self.coords[:,0] += self.centre[0]
            self.coords[:,1] += self.centre[1]
        else:
            x = self.centre[0]-about[0]
            y = self.centre[1]-about[1]
            r = np.sqrt(x**2+y**2)
            if r == 0:
                pass
            else:
                theta_total = np.arcsin(x/r)+phi
                self.coords[:,0] += r*np.sin(theta_total)
                self.coords[:,1] += r*np.cos(theta_total)
        if about is None:
            for hole in self.holes:
                hole.rotate_coords(phi, self.centre)
            for spiral in self.spirals:
                spiral.rotate_coords(phi, self.centre)
        else:
            for hole in self.holes:
                hole.rotate_coords(phi, about)
            for spiral in self.spirals:
                spiral.rotate_coords(phi, about)
        
    def plot(self, ax, color="#1f77b4"):
        ax.plot(self.coords[:,0], self.coords[:,1], color)
        for hole in self.holes:
            hole.plot(ax, color)
        for spiral in self.spirals:
            spiral.plot(ax, color)
        ax.set_aspect('equal', 'box')
#        r = self._r
#        ax.set_xlim(-1.1*r,1.1*r)
#        ax.set_ylim(-1.1*r,1.2*r)
        
    def project(self, n, phi=None, about=None, lower=None, upper=None, background=0, noise=None, gauss=None):
        if lower is None:
            lower = np.min(self.coords[:,0])
        if upper is None:
            upper = np.max(self.coords[:,0])
        if phi is not None:
            self.rotate_coords(phi, about)
        t = [[] for _ in range(n)]
        n -= 1
        xs = [i*(upper-lower)/n+(lower-upper)/2 for i in range(n+1)]
        for a,b in zip(self.coords[:-1],self.coords[1:]):
            dif = b-a
            for i,x0 in enumerate(xs):
                if a[0] <= x0 < b[0] or b[0] < x0 <= a[0]:
                    t1 = a[1] + dif[1]*(x0-a[0])/dif[0]
                    t[i].append(t1)
        for x in t:
            x.sort()
        t = np.array([sum(a-b for a,b in zip(i[1::2],i[0::2])) for i in t])*self.density
        
        for hole in self.holes:
            t += hole.project(n+1, None, about, lower, upper)
        
        for spiral in self.spirals:
            t += spiral.project(n+1, None, about, lower, upper)
        
        #Experimental Noise
        t += background
        if gauss:
            t = gaussian_filter(t, sigma=gauss)
        if noise:
            t += np.random.normal(0, noise*t)
        return t

class Spiral(Shape):
    def __init__(self, r=1, strtness=.04, thckness=(.01, .1), clock=True, n=360, dtheta=0., centre=(0,0), density=1):
        self._theta = r/strtness/n
        theta = [self._theta*i for i in range(n+1)]
        if clock:
            self.theta = np.array(theta+theta[::-1]+[theta[0]]) + dtheta
        else:
            self.theta = -np.array(theta+theta[::-1]+[theta[0]]) + dtheta
        t0 = thckness[0]+strtness
        self.r = np.array([strtness*t for t in theta]
                +[min(t*t0, strtness*t+thckness[1]) for t in theta[::-1]]
                +[strtness*theta[0]])
        super().__init__(-1, n, centre, density=density)
    
if __name__ == '__main__':
    
    #shape = Shape(np.random.normal(0.8,.1), 120, sig=.03, centre=(np.random.normal(0,.3),np.random.normal(0,.3)), h_centre=np.random.rand(2)*.8-.4, h_r=np.random.normal(.1,.02))
    #shape = Shape(np.random.normal(1,.1), 90, sig=.0, centre=(np.random.normal(0,.3),np.random.normal(0,.3)), h_centre=np.random.rand(2)*.8-.4, h_r=np.random.normal(.1,.02))
    shape = Shape(np.random.normal(1,.1), 90, sig=.0,
                  centre=(np.random.normal(0,.3),np.random.normal(0,.3)),
                  s_centre=np.random.rand(2)*.3-.15, s_r=np.random.normal(.6,.02),
                  s_s=.04, s_t=(.01,.1), s_clock=False, s_theta=np.random.rand(1)*np.pi*2)
#    shape = Spiral(r=.8, strtness=.04, thckness=(.01, .1),
#                   clock=True, n=90, dtheta=0.,
#                   centre=(np.random.normal(0,.3),np.random.normal(0,.3)))
    plt.close("all")
    fig, axs = plt.subplots(2,2)
    ax = axs[0, 0]
    shape.plot(ax)
    ax.set(xlim=(-1.5,1.5), ylim=(-1.5,1.5))
    ax.set_aspect('equal', 'box')
    
    ax = axs[0,1]
    shape.rotate_coords(np.pi/4, (.0,.0))
    shape.plot(ax, "k")
    ax.set(xlim=(-1.5,1.5), ylim=(-1.5,1.5))
    ax.set_aspect('equal', 'box')
#    
#    shape.rotate_coords(np.pi)
#    shape.plot(ax, "r")
#    
#    shape.rotate_coords(3*np.pi/2)
#    shape.plot(ax, "g")
    
    n = 100
    upper = 1.5
    lower = -1.5
    xs = [i*(upper-lower)/n+(lower-upper)/2 for i in range(n)]
    
    t = shape.project(n, 0.0, about=(.0,.0), lower=lower, upper=upper, background=20, noise=0.001, gauss=0.5)
    ax = axs[1,0]
    ax.plot(xs, t)
    ax.set(xlim=(lower,upper))
    
    t = shape.project(n,  np.pi/4, about=(.0,.0), lower=lower, upper=upper, background=20, noise=0.001, gauss=0.5)
    ax = axs[1,1]
    ax.plot(xs, t, "k")
    ax.set(xlim=(lower,upper))
    