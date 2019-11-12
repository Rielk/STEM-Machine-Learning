# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 18:44:28 2019

@author: William
"""
from shape import Shape
import numpy as np

def shapes_gen(count, r, sig_r, n, centre, sig_c, sig, h_range, h_min, h_r, sig_hr, h_n=None):
    return np.array([Shape(np.random.normal(r,sig_r), n, (np.random.normal(centre[0],sig_c[0]),
                                            np.random.normal(centre[1],sig_c[1])), sig, 
    (np.random.rand()*h_range[0]-h_min[0],np.random.rand()*h_range[1]-h_min[1]), 
    np.random.normal(h_r,sig_hr))for _ in range(count)])


def projection_gen(shapes, n, angles, sig_as, about, lower, upper, background, noise, gauss):
    return np.array([[shape.project(n, np.random.normal(angle,sig_a), about, lower, upper, background, noise, gauss) for angle, sig_a in zip(angles, sig_as)]for shape in shapes])


if __name__ == '__main__':
    r = 1
    sig_r = .1
    n = 120
    centre = (0,0)
    sig_c = (.3,.3)
    sig = .03
    h_range = (.8,.8)
    h_min = (.4,.4)
    h_r = .1
    sig_hr = .02
    count = 1000
    shapes = shapes_gen(count, r,sig_r,n,centre,sig_c,sig,h_range,h_min,h_r, sig_hr)
    
    n = 32
    angles = (0,np.pi/4)
    sig_as = (.02, .02)
    about = (0.,0.)
    lower = -1.5
    upper = 1.5
    background = 20
    noise=0.001
    gauss=0.5
    projections = projection_gen(shapes, n, angles, sig_as, about, lower, upper, background, noise, gauss)
    