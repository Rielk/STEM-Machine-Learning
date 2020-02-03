#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 19:59:44 2019

@author: william
"""
from models import Adam
import matplotlib.pyplot as plt
from runner import run
import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
plt.close("all")

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
               "angles":(0., np.pi/4),
               "sig_a":.0,
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
            "angles":(0., np.pi/4),
            "sig_a":.0,
            "about":(0.,0.),
            "lower":-1.5,
            "upper":1.5,
            "background":20.,
            "gauss":.5,
            "noise":.001,
            }

train_params = {"model":Adam, "epochs":200, "verification":True, "patience":10, "restore_best_weights":True}

#Make the model
net_structure = [32 for _ in range(8)]
net_structure += [0]
net_structure += [64 for _ in range(4)]
net_structure += [32 for _ in range(4)]
model_params = {"model":Adam,
            "input_points":(32,2),
            "node_per_layer":net_structure,
            "dropout_rate":.1,
            "max_norm":3.,
            "layer_activation":"relu",
            "output_activation":"softmax",
            "optimizer":"rmsprop",#sgd
            "loss":"categorical_crossentropy",
            }

optimizers = ["RMSprop"]#["sgd", "RMSprop", "Adagrad", "Adadelta", "Adam", "Adamax", "Nadam"]

run_data = []
for o in optimizers:
    print("\nBegining tests with optimizer: {}".format(o))
    model_params["optimizer"] = o
    accuracy = 0
    n = 0
    while accuracy < 0.6 and n <= 5:
        model, accuracy, epoch, t = run(train_params, model_params, data_params, v_params, True)
        #del model
    run_data.append((accuracy, epoch, t, n))
