# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 18:44:28 2019

@author: William
"""
from shape import Shape
import numpy as np
import matplotlib.pyplot as plt

def shapes_gen(count, r, sig_r, n, centre, sig_c, sig, h_range, h_r, sig_hr, h_n=None):
    h_theta = np.random.rand()*2*np.pi
    h_rad = h_range*np.random.rand()
    h_centre = (h_rad*np.sin(h_theta),h_rad*np.cos(h_theta))
    return np.array([Shape(np.random.normal(r,sig_r), n,
                           (np.random.normal(centre[0],sig_c[0]),
                            np.random.normal(centre[1],sig_c[1])),
                            sig, h_centre, np.random.normal(h_r,sig_hr))
                           for _ in range(count)])


def projection_gen(shapes, n, angles, sig_as, about, lower, upper, background, noise, gauss):
    return np.array([[shape.project(n, np.random.normal(angle,sig_a), about, lower, upper, background, noise, gauss) for angle, sig_a in zip(angles, sig_as)]for shape in shapes])


if __name__ == '__main__':
    r = 1.                      #Base radius of shape
    sig_r = .1                  #Variance on average radius
    angle_count = 120           #Number of radius points in a shape
    centre = (0,0)              #Average centre of shape
    sig_c = (.3,.3)             #Variance on the coordinates of centre
    sig = .03                   #The variance on radius points
    h_range = .7                #The maximum distance of hole from centre
    h_r = .1                    #Base radius of hole
    sig_hr = .02                #Variance on average radius of hole
    shape_count = 100           #Number of shapes to create
    
    data_points = 32            #Number of points on projection
    angles = (0,np.pi/4)        #Approximate angle to evaluate projection at
    sig_as = (.02, .02)         #Variance on each angle projection is taken
    about = (0.,0.)             #The point that rotation occurs about
    lower = -1.5                #Lower limit on projection range
    upper = 1.5                 #Upper limit on projection range
    background = 20             #Additional background value
    gauss=0.5                   #The amount of gaussian blur    
    noise=0.001                 #Ratio of t to make the variance on
                                #projection data (eg, 0.1, results in each
                                #data point having a varianve of 0.1t)
    
    shapes = shapes_gen(shape_count, r,sig_r,angle_count,centre,sig_c,sig,h_range,h_r, sig_hr)
    projections = projection_gen(shapes, data_points, angles, sig_as, about, lower, upper, background, noise, gauss)

    plt.close("all")    
    fig, axs = plt.subplots(2,2)
    shapes[0].plot(axs[0,0])
    shapes[1].plot(axs[1,0])
    shapes[2].plot(axs[0,1])
    shapes[3].plot(axs[1,1])    