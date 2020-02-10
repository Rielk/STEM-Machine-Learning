# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 18:44:28 2019

@author: William
"""
from shape import Shape
import numpy as np
import matplotlib.pyplot as plt
import keras

def shapes_gen(count, r, sig_r, n, centre, sig_c, sig,
               h_range=None, h_r=None, sig_hr=None,
               s_r=None, s_range=None, s_n=None, s_density=None,
               sig_sd=None, s_s=None, sig_ss=None, s_t=None,
               sig_st=None, s_clock=None):
    if h_range is not None and s_r is not None:
        raise ValueError("Do not provide both hole and spiral parameters")
    elif h_range is not None:
        h_theta = np.random.rand()*2*np.pi
        h_rad = h_range*np.random.rand()
        h_centre = (h_rad*np.sin(h_theta),h_rad*np.cos(h_theta))
        return np.array([Shape(np.random.normal(r,sig_r), n,
                               (np.random.normal(centre[0],sig_c[0]),
                                np.random.normal(centre[1],sig_c[1])),
                                sig, h_centre, np.random.normal(h_r,sig_hr))
                                for _ in range(count)])
    elif s_r is not None:
        s_theta = np.random.rand()*2*np.pi
        s_rad = s_range*np.random.rand()
        s_centre = (s_rad*np.sin(s_theta),s_rad*np.cos(s_theta))
        return np.array([Shape(np.random.normal(r, sig_r), n,
                               (np.random.normal(centre[0],sig_c[0]),
                                np.random.normal(centre[1],sig_c[1])),
                                sig, s_centre=s_centre, s_n=s_n,
                                s_density=np.random.normal(s_density, sig_sd),
                                s_clock=s_clock,
                                s_s=np.random.normal(s_s, sig_ss),
                                s_t=tuple(np.random.normal(s_t, sig_st)),
                                s_theta=np.random.rand()*np.pi*2) for _ in range(count)])
    else:
        return np.array([Shape(np.random.normal(r,sig_r), n,
                           (np.random.normal(centre[0],sig_c[0]),
                            np.random.normal(centre[1],sig_c[1])),
                            sig) for _ in range(count)])


def projection_gen(shapes, n, angles, sig_a, about, lower, upper, background, noise, gauss):
    try:
        n[0]
    except (IndexError,TypeError):
        n = tuple(n for _ in angles)
    return np.array([[shape.project(x, np.random.normal(angle,sig_a),
                                        about, lower, upper, background,
                                        noise, gauss)
                                        for x, angle in zip(n, angles)]
                                        for shape in shapes])

def data_gen(params):
    """
    Params is a dictionary with key corresponding to values:
    data_count_h    -int        -Total count of the data set which have holes
    data_count_s    -int        -Total count of the data set which are solid

    r               -float      -Base radius of shape
    sig_r           -float      -Variance on average radius
    angle_count     -int        -Number of radius points in a shape
    centre          -iter       -Average centre of shape
    sig_c           -iter       -Variance on the coordinates of centre
    sig             -float      -The variance on radius points
    h_range         -float      -The maximum distance of hole from centre
    h_r             -float      -Base radius of hole
    sig_hr          -float      -Variance on average radius of hole
    
    data_points     -int/iter   -Number of points on projection(can specify for each angle)
    angles          -iter       -Approximate angle to evaluate projection at
    sig_as          -float      -Variance on each angle projection is taken
    about           -float      -The point that rotation occurs about
    lower           -float      -Lower limit on projection range
    upper           -float      -Upper limit on projection range
    background      -float      -Additional background value
    gauss           -float      -The amount of gaussian blur    
    noise           -float      -Ratio of t to make the variance on
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
    
    try:
        h_range = params["h_range"]
        h_r = params["h_r"]
        sig_hr = params["sig_hr"]
        use_holes = True
    except NameError:
        use_holes = False
    
    try:
        s_r = params["s_r"]
        s_range = params["s_range"]
        try:
            s_n = params["s_n"]
        except NameError:
            s_n = angle_count
        s_density = params["s_density"]
        sig_sd = params["sig_sd"]
        s_s = params["s_s"]
        sig_ss = params["sig_ss"]
        s_t = params["s_t"]
        sig_st = params["sig_st"]
        s_clock = params["s_clock"]
    except NameError:
        use_spirals = False
    
    data_points = params["data_points"]
    angles = params["angles"]
    sig_a = params["sig_a"]
    about = params["about"]
    lower = params["lower"]
    upper = params["upper"]
    background = params["background"]
    gauss = params["gauss"]
    noise = params["noise"]
    
    if use_holes and use_spirals:
        raise ValueError("Do not provide both hole and spiral parameters")
    elif use_holes:
        #Shapes with holes
        print("Generating shapes with holes")
        shapes = shapes_gen(shape_count_h, r,sig_r,angle_count,centre,sig_c,sig,h_range,h_r, sig_hr)
        #Shapes without holes
        print("Generating shapes without holes")
        shapes = np.append(shapes, shapes_gen(shape_count_s, r,sig_r,angle_count,centre,sig_c,sig))
    elif use_spirals:
        print("Generating clockwise spirals")
        shapes = shapes_gen(shape_count_h, r,sig_r,angle_count,centre,
                            s_r=None, s_range=None, s_n=None, s_density=None,
                            sig_sd=None, s_s=None, sig_ss=None, s_t=None,
                            sig_st=None, s_clock=None)
        print("Generating anticlockwise spirals")
        shapes = np.append(shapes, shapes_gen(shape_count_s, r,sig_r,angle_count,centre,
                                              s_r=None, s_range=None, s_n=None, s_density=None,
                                              sig_sd=None, s_s=None, sig_ss=None, s_t=None,
                                              sig_st=None, s_clock=None))
    else:
        raise ValueError("provide either hole or spiral parameters")
    #Make the projections
    print("Generating projections")
    projections = projection_gen(shapes, data_points, angles, sig_a, about, lower, upper, background, noise, gauss)
    #Normalise projections
    minimum = np.inf
    for i in range(projections.shape[0]):
        for j in range(projections.shape[1]):
            mi = np.min(projections[i,j])
            if mi < minimum:
                minimum = mi
    projections -= mi
    
    maximum = -np.inf
    for i in range(projections.shape[0]):
        for j in range(projections.shape[1]):
            ma = np.max(projections[i,j])
            if ma > maximum:
                maximum = ma
    projections /= maximum
    # Plit the projections into lists
    projections = [projections[:,i] for i in range(len(angles))]
    #Produce the traget, 1 for holes, and 0 for solids
    print("Generating labels")
    labels = keras.utils.to_categorical(
            np.array([[1] if x < shape_count_h else [0] 
                        for x in range(shape_count_h+shape_count_s)]),
                        num_classes=2)
    print("Finished data generation\n")
    return shapes, np.array(projections), np.array(labels)

def data_default_params(n=200):
    """
    Produces a 'reasonable' set of parameters in a dictionary
    """
    return {
            "data_count_h":int(n/2),
            "data_count_s":int(n/2),
    
            "r":1.,    
            "sig_r":.1,
            "angle_count":90,
            "centre":(0,0),
            "sig_c":(.3,.3),
            "sig":.03,
            "h_range":.7,
            "h_r":.1,
            "sig_hr":.02,
            
            "data_points":32,
            "angles":(0,np.pi/4),
            "sig_a":.02,
            "about":(0.,0.),
            "lower":-1.5,
            "upper":1.5,
            "background":20.,
            "gauss":.5,
            "noise":.001,
            }

def request_data_params_format():
    print("""data_count_h    -int        
    -Total count of the data set which have holes
          
data_count_s    -int        
    -Total count of the data set which are solid

r               -float
    -Base radius of shape

sig_r           -float      
    -Variance on average radius

angle_count     -int        
    -Number of radius points in a shape

centre          -iter       
    -Average centre of shape

sig_c           -iter       
    -Variance on the coordinates of centre

sig             -float      
    -The variance on radius points

h_range         -float      
    -The maximum distance of hole from centre

h_r             -float      
    -Base radius of hole

sig_hr          -float      
    -Variance on average radius of hole
    
data_points     -int/iter   
    -Number of points on projection(can specify for each angle)

angles          -iter       
    -Approximate angle to evaluate projection at

sig_a          -float      
    -Variance on each angle projection is taken

about           -float      
    -The point that rotation occurs about

lower           -float      
    -Lower limit on projection range

upper           -float      
    -Upper limit on projection range

background      -float      
    -Additional background value

gauss           -float      
    -The amount of gaussian blur    

noise           -float      
    -Ratio of t to make the variance on projection data (eg, 0.1, results in each data point having a varianve of 0.1t)
    """)

if __name__ == '__main__':
    params = data_default_params(20)
    params["channels"] = 1
    
    shapes, projections, labels = data_gen(params)

    plt.close("all")    
    fig, axs = plt.subplots(2,2)
    shapes[0].plot(axs[0,0])
    shapes[1].plot(axs[1,0])
    shapes[params["data_count_h"]].plot(axs[0,1])
    shapes[params["data_count_h"]+1].plot(axs[1,1])
    
    plt.figure()
    plt.plot(projections[0][0])
    plt.figure()
    plt.plot(projections[0][params["data_count_h"]])
    