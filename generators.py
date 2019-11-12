# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 18:44:28 2019

@author: William
"""
from shape import Shape
import numpy as np

def shapes_gen(count, r, sig_r, n, centre, sig_c, sig, h_range, h_min, h_r, sig_hr, h_n=None):
    shapes = np.array([Shape(np.random.normal(r,sig_r), n, (np.random.normal(centre[0],sig_c[0]),np.random.normal(centre[1],sig_c[1])), sig, (np.random.rand()*h_range[0]-h_min[0],np.random.rand()*h_range[1]-h_min[1]), np.random.normal(h_r,sig_hr))for _ in range(count)])
    return shapes


if __name__ == '__main__':
    r = 1
    sig_r = .1
    n = 120
    centre = (0,0)
    sig_c = (.3,.3)
    sig = .05
    h_range = (.8,.8)
    h_min = (.4,.4)
    h_r = .1
    sig_hr = .02
    count = 60000
    shapes = shapes_gen(count, r,sig_r,n,centre,sig_c,sig,h_range,h_min,h_r, sig_hr)
    
    