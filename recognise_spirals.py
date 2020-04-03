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
data_params = {"data_count_c":30000,
               "data_count_a":30000,
               #"data_count_n":0,
               
               "r":1.,    
               "sig_r":.1,
               "angle_count":90,
               "centre":(0,0),
               "sig_c":(.0,.0),#(.3,.3),
               "sig":.05,
               "s_range":.1,
               "s_r":.5,
               "sig_sr":.05,
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
               "background":0.,#20.,
               "gauss":0.,#.5,
               "noise":0.,#.001,
               "limit_var":.05
               }

v_params = data_params.copy()
v_params["data_count_c"] = 10000
v_params["data_count_a"] = 10000

train_params = {"model":Grant,
                "epochs":100,
                "verification":True,
                "patience":10,
                "restore_best_weights":True
                }
#
#Make the model
#net_structure = [64 for _ in range(2)]
net_structure = [-1]
net_structure += [(32,4) for _ in range(3)]
net_structure += [32 for _ in range(2)]
net_structure += [0]
#net_structure += [64 for _ in range(2)]
#net_structure += [32 for _ in range(4)]
net_structure += [16 for _ in range(4)]
#net_structure = [64 for _ in range(4)]
#net_structure += [(64,4) for _ in range(4)]
#net_structure += [0]
##net_structure += [(32,4) for _ in range(2)]
##net_structure += [0]
#net_structure += [64 for _ in range(2)]
#net_structure += [32 for _ in range(4)]
#net_structure += [16 for _ in range(4)]
#net_structure += [0]
model_params = {"model":Grant,
            "input_points":(64,2),
            "output_categories":2,
            "node_per_layer":net_structure,
            "dropout_rate":.1,
            "max_norm":3.,
            "layer_activation":"relu",
            "output_activation":"softmax",
            "optimizer":"Adadelta",
            "loss":"categorical_crossentropy",
            }

n = 0
accuracy = 0
while n < 3 and accuracy <= .65:
    model, accuracy, epoch, t, keys = run(train_params, model_params, data_params, v_params, True)
    n += 1

if __name__ == '__main__':
    print(keys)
    v_params["data_count_c"] = 1000
    v_params["data_count_a"] = 1000
    v_params["data_count_n"] = 0
    shapes, data, labels = data_gen(v_params)
