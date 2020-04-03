# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 00:32:55 2020

@author: William
"""
from models import Grant
from generators import data_gen
import matplotlib.pyplot as plt
from runner import run
import numpy as np
plt.close("all")

#Produce learning data
data_params = {"data_count_h":3000,
               "data_count_s":3000,
               
               "r":1.,    
               "sig_r":.1,
               "angle_count":90,
               "centre":(0,0),
               "sig_c":(.0,.0),
               "sig":.05,
               "h_range":.4,
               "h_r":.05,
               "sig_hr":.005,
               
               "data_points":64,
               "angles":[np.pi*i/64 for i in range(64)],
               #"angles":(0.,np.pi/2),
               "sig_a":.0,
               "about":(0.,0.),
               "lower":-1.5,
               "upper":1.5,
               "background":20.,
               "gauss":.5,
               "noise":.001,
               "limit_var":.05
               }

v_params = data_params.copy()
v_params["data_count_h"] = 1000
v_params["data_count_s"] = 1000

train_params = {"model":Grant,
                "epochs":100,
                "verification":True,
                "patience":10,
                "restore_best_weights":True
                }

##Make the model
#net_structure = [0]
#net_structure += [(32,4) for _ in range(4)]
##net_structure += [96 for _ in range(2)]
#net_structure += [64 for _ in range(10)]
##net_structure += [16 for _ in range(4)]

#net_structure = [64 for _ in range(4)]
net_structure = [-1]
net_structure += [(32,4,2) for _ in range(3)]
net_structure += [32 for _ in range(2)]
net_structure += [0]
#net_structure += [64 for _ in range(4)]
#net_structure += [32 for _ in range(4)]
net_structure += [16 for _ in range(4)]
model_params = {"model":Grant,
            "input_points":(64,2),
            "output_categories":2,
            "node_per_layer":net_structure,
            "dropout_rate":.1,
            "max_norm":3.,
            "layer_activation":"relu",
            "output_activation":"softmax",
            "optimizer":"AdaDelta",
            "loss":"categorical_crossentropy",
            }

model, accuracy, epoch, t, keys = run(train_params, model_params, data_params, v_params, True)

print(keys)

v_params["data_count_h"] = 100
v_params["data_count_s"] = 100
shapes, data, labels = data_gen(v_params)
