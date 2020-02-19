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
data_params = {"data_count_h":30000,
               "data_count_s":30000,
               
               "r":1.,    
               "sig_r":.1,
               "angle_count":90,
               "centre":(0,0),
               "sig_c":(.0,.0),#(.3,.3),
               "sig":.05,
               "s_range":.1,
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
               "sig_a":.0,
               "about":(0.,0.),
               "lower":-1.5,
               "upper":1.5,
               "background":20.,
               "gauss":.5,
               "noise":.001,
               #"limit_var":.0
               }

v_params = data_params.copy()
v_params["data_count_h"] = 10000
v_params["data_count_s"] = 10000

train_params = {"model":Brian,
                "epochs":200,
                "verification":True,
                "patience":10,
                "restore_best_weights":True
                }

#Make the model
net_structure = [(64,4) for _ in range(4)]
net_structure += [0]
net_structure += [64 for _ in range(2)]
net_structure += [32 for _ in range(4)]
net_structure += [16 for _ in range(4)]
model_params = {"model":Brian,
            "input_points":(64,2),
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
while n <= 0.7 and accuracy < 10:
    model, accuracy, epoch, t = run(train_params, model_params, data_params, v_params, True)

v_params["data_count_h"] = 100
v_params["data_count_s"] = 100
shapes, data, labels = data_gen(v_params)
