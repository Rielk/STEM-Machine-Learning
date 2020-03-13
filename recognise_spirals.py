# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 00:32:55 2020

@author: William
"""
from models import Brian
from generators import data_gen
import matplotlib.pyplot as plt
from runner import run
import numpy as np
plt.close("all")

#Produce learning data
data_params = {"data_count_c":30000,
               "data_count_a":30000,
               "data_count_n":0,
               
               "r":1.,    
               "sig_r":.1,
               "angle_count":90,
               "centre":(0,0),
               "sig_c":(.0,.0),#(.3,.3),
               "sig":.05,
               "s_range":.05,
               "s_r":.6,
               "sig_sr":.02,
               "s_density":1.,
               "sig_sd":.0,
               "s_s":.04,
               "sig_ss":.008,
               "s_t":(.01,.1),
               "sig_st":(.002,.02),
               
               "data_points":64,
               "angles":(0., np.pi/2),
               "sig_a":.01,
               "about":(0.,0.),
               "lower":-1.5,
               "upper":1.5,
               "background":20.,
               "gauss":.5,
               "noise":.001,
               "limit_var":.05
               }

v_params = data_params.copy()
v_params["data_count_c"] = 10000
v_params["data_count_a"] = 10000

train_params = {"model":Brian,
                "epochs":200,
                "verification":True,
                "patience":40,
                "restore_best_weights":True
                }

#Make the model
net_structure = [64 for _ in range(4)]
net_structure += [(64,4) for _ in range(4)]
net_structure += [0]
net_structure += [64 for _ in range(2)]
net_structure += [32 for _ in range(4)]
net_structure += [16 for _ in range(4)]
model_params = {"model":Brian,
            "input_points":(64,2),
            "output_categories":3,
            "node_per_layer":net_structure,
            "dropout_rate":.1,
            "max_norm":3.,
            "layer_activation":"relu",
            "output_activation":"softmax",
            "optimizer":"Adagrad",
            "loss":"categorical_crossentropy",
            }

n = 0
accuracy = 0
while n < 3 and accuracy <= .65:
    model, accuracy, epoch, t = run(train_params, model_params, data_params, v_params, True)
    n += 1

v_params["data_count_c"] = 100
v_params["data_count_a"] = 100
shapes, data, labels = data_gen(v_params)
