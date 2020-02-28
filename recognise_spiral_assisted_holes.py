# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 17:56:31 2020

@author: William
"""
from models import Charlie
from generators import data_gen
import matplotlib.pyplot as plt
import keras.backend as KB
import keras.losses as KL
from runner import run
import numpy as np
plt.close("all")

#Produce learning data
data_params = {"data_count_h":150,
               "data_count_s":150,
               "data_count_c":100,
               "data_count_a":100,
               "data_count_n":0,
               
               "r":1.,    
               "sig_r":.1,
               "angle_count":90,
               "centre":(0,0),
               "sig_c":(.3,.3),
               "sig":.05,
               "h_range":.4,
               "h_r":.075,
               "sig_hr":.0075,
               "s_range":.0,
               "s_r":.6,
               "sig_sr":.02,
               "s_density":1.,
               "sig_sd":.0,
               "s_s":.04,
               "sig_ss":.004,
               "s_t":(.01,.1),
               "sig_st":(.001,.01),
               
               "data_points":64,
               "angles":(0., np.pi/2),
               "sig_a":.0,
               "about":(0.,0.),
               "lower":-1.5,
               "upper":1.5,
               "background":20.,
               "gauss":.5,
               "noise":.001,
               "limit_var":.0
               }

v_params = data_params.copy()
v_params["data_count_h"] = 5000
v_params["data_count_s"] = 5000
v_params["data_count_c"] = 1
v_params["data_count_a"] = 1
v_params["data_count_n"] = 0

train_params = {"model":Charlie,
                "epochs":1000,
                "verification":True,
                "patience":20,
                "restore_best_weights":True
                }

#Make the model
net_structure = [64 for _ in range(4)]
net_structure += [(64,4) for _ in range(4)]
net_structure += [0]
net_structure += [64 for _ in range(2)]
net_structure += [32 for _ in range(4)]
net_structure += [16 for _ in range(4)]

def assister_loss(y_true, y_pred):
    return KB.in_train_phase(KL.categorical_crossentropy(y_true, y_pred),
                             KB.zeros_like(KL.categorical_crossentropy(y_true, y_pred)))
KB.zeros_like

model_params = {"model":Charlie,
            "input_points":(64,2),
            "output_categories":(2,3),
            "weight_ratio":[.0,1.],
            "node_per_layer":net_structure,
            "dropout_rate":.1,
            "max_norm":3.,
            "layer_activation":"relu",
            "output_activation":"softmax",
            "optimizer":"Adam",
            "loss":"categorical_crossentropy",
            "a_loss":assister_loss
            }

n = 0
accuracy = 0
while n < 1 and accuracy <= .65:
    model, accuracy, epoch, t = run(train_params, model_params, data_params, v_params, True)
    n += 1

v_params = data_params.copy()
v_params["data_count_h"] = 50
v_params["data_count_s"] = 50
v_params["data_count_c"] = 1
v_params["data_count_a"] = 1
v_params["data_count_n"] = 0
shapes, data, labels = data_gen(v_params)