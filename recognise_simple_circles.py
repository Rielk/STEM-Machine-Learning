#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 19:59:44 2019

@author: william
"""
from models import Adam, Adam_default_params
import matplotlib.pyplot as plt
from runner import run
import numpy as np

plt.close("all")

#Make the model
model_params = Adam_default_params()
model_params["input_points"] = (32,2)

#Produce learning data
data_params = {"data_count_h":30000,
               "data_count_s":30000,
               
               "r":1.,    
               "sig_r":.1,
               "angle_count":90,
               "centre":(0,0),
               "sig_c":(.3,.3),
               "sig":.0,
               "h_range":.4,
               "h_r":.2,
               "sig_hr":.02,
               
               "data_points":32,
               "angles":(0, np.pi/4),
               "sig_as":(.02, .02),
               "about":(0.,0.),
               "lower":-1.5,
               "upper":1.5,
               "background":20.,
               "gauss":.5,
               "noise":.001,
               }

v_params = {"data_count_h":10000,
            "data_count_s":10000,
            
            "r":1.,    
            "sig_r":.1,
            "angle_count":90,
            "centre":(0,0),
            "sig_c":(.3,.3),
            "sig":.0,
            "h_range":.4,
            "h_r":.2,
            "sig_hr":.02,
            
            "data_points":32,
            "angles":(0, np.pi/4),
            "sig_as":(.04, .04),
            "about":(0.,0.),
            "lower":-1.5,
            "upper":1.5,
            "background":20.,
            "gauss":.5,
            "noise":.001,
            }

train_params = {"model":Adam, "epochs":100, "verification":True}

run(train_params, model_params, data_params, v_params, True)
