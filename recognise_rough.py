# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 00:32:55 2020

@author: William
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
               "sig":.03,
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
            "sig":.03,
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
            "input_points":(32,3),
            "node_per_layer":net_structure,
            "dropout_rate":.1,
            "max_norm":3.,
            "layer_activation":"relu",
            "output_activation":"softmax",
            "optimizer":"Adagrad",
            "loss":"categorical_crossentropy",
            }

model, accuracy, epoch, t = run(train_params, model_params, data_params, v_params, True)
