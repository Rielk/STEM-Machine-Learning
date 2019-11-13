# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 18:44:28 2019

@author: William
"""
from shape import Shape
import numpy as np
import matplotlib.pyplot as plt

def shapes_gen(count, r, sig_r, n, centre, sig_c, sig, h_range=None, h_r=None, sig_hr=None, h_n=None):
    if h_range is not None:
        h_theta = np.random.rand()*2*np.pi
        h_rad = h_range*np.random.rand()
        h_centre = (h_rad*np.sin(h_theta),h_rad*np.cos(h_theta))
        return np.array([Shape(np.random.normal(r,sig_r), n,
                           (np.random.normal(centre[0],sig_c[0]),
                            np.random.normal(centre[1],sig_c[1])),
                            sig, h_centre, np.random.normal(h_r,sig_hr))
                            for _ in range(count)])
    else:
        return np.array([Shape(np.random.normal(r,sig_r), n,
                           (np.random.normal(centre[0],sig_c[0]),
                            np.random.normal(centre[1],sig_c[1])),
                            sig) for _ in range(count)])


def projection_gen(shapes, n, angles, sig_as, about, lower, upper, background, noise, gauss):
    return np.array([[shape.project(n, np.random.normal(angle,sig_a),
                                    about, lower, upper, background,
                                    noise, gauss)
                                    for angle, sig_a in zip(angles, sig_as)]
                                    for shape in shapes])

def data_gen(params):
    """
    Params is a dictionary with key corresponding to values:
    data_count_h    -int    -Total count of the data set which have holes
    data_count_s    -int    -Total count of the data set which are solid

    r               -float  -Base radius of shape
    sig_r           -float  -Variance on average radius
    angle_count     -int    -Number of radius points in a shape
    centre          -tuple  -Average centre of shape
    sig_c           -tuple  -Variance on the coordinates of centre
    sig             -float  -The variance on radius points
    h_range         -float  -The maximum distance of hole from centre
    h_r             -float  -Base radius of hole
    sig_hr          -float  -Variance on average radius of hole
    
    data_points     -int    -Number of points on projection
    angles          -tuple  -Approximate angle to evaluate projection at
    sig_as          -float  -Variance on each angle projection is taken
    about           -float  -The point that rotation occurs about
    lower           -float  -Lower limit on projection range
    upper           -float  -Upper limit on projection range
    background      -float  -Additional background value
    gauss           -float  -The amount of gaussian blur    
    noise           -float  -Ratio of t to make the variance on
                            projection data (eg, 0.1, results in each
                            data point having a varianve of 0.1t)
    """
    shape_count_h = params["data_count_h"]
    shape_count_s = params["data_count_s"]
    
    r = params["r"]
    sig_r = params["sig_r"]
    angle_count = params["angle_count"]
    centre = params["centre"]
    sig_c = params["sig_c"]
    sig = params["sig"]
    h_range = params["h_range"]
    h_r = params["h_r"]
    sig_hr = params["sig_hr"]
    
    data_points = params["data_points"]
    angles = params["angles"]
    sig_as = params["sig_as"]
    about = params["about"]
    lower = params["lower"]
    upper = params["upper"]
    background = params["background"]
    gauss = params["gauss"]
    noise = params["noise"]
    
    #Shapes with holes
    shapes = shapes_gen(shape_count_h, r,sig_r,angle_count,centre,sig_c,sig,h_range,h_r, sig_hr)
    #Shapes without holes
    shapes = np.append(shapes, shapes_gen(shape_count_s, r,sig_r,angle_count,centre,sig_c,sig))
    #Make the projections
    projections = projection_gen(shapes, data_points, angles, sig_as, about, lower, upper, background, noise, gauss)
    return shapes, projections

if __name__ == '__main__':
    params = {
            "data_count_h":2,
            "data_count_s":2,
    
            "r":1.,    
            "sig_r":.1,
            "angle_count":120,
            "centre":(0,0),
            "sig_c":(.3,.3),
            "sig":.03,
            "h_range":.7,
            "h_r":.1,
            "sig_hr":.02,
            
            "data_points":32,
            "angles":(0,np.pi/4),
            "sig_as":(.02, .02),
            "about":(0.,0.),
            "lower":-1.5,
            "upper":1.5,
            "background":20.,
            "gauss":.5,
            "noise":.001,
            }
    
    shapes, projections = data_gen(params)

    plt.close("all")    
    fig, axs = plt.subplots(2,2)
    shapes[0].plot(axs[0,0])
    shapes[1].plot(axs[1,0])
    shapes[2].plot(axs[0,1])
    shapes[3].plot(axs[1,1])    