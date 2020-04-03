# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 17:56:31 2020

@author: William
"""
from models import Charlie, Charlie_recompile
from generators import data_gen
import matplotlib.pyplot as plt
import keras.backend as KB
import keras.losses as KL
from runner import run
import numpy as np
plt.close("all")

#Produce learning data
data_params = {"data_count_h":0,
               "data_count_s":30000,
               "data_count_c":1,
               "data_count_a":1,
               #"data_count_n":0,
               
               "r":1.,    
               "sig_r":.1,
               "angle_count":90,
               "centre":(0,0),
               "sig_c":(.0,.0),
               "sig":.05,
               "h_range":.4,
               "h_r":.1,
               "sig_hr":.01,
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
               "background":20.,
               "gauss":.5,
               "noise":.001,
               "limit_var":.05,
               
               "output_number":2
               }

v_params = data_params.copy()
v_params["data_count_h"] = 0
v_params["data_count_s"] = 10000
v_params["data_count_c"] = 1
v_params["data_count_a"] = 1
#v_params["data_count_n"] = 0

train_params = {"model":Charlie,
                "epochs":100,
                "verification":True,
                "patience":10,
                "restore_best_weights":True,
                }

#Make the model
net_structure = [(32,4) for _ in range(3)]
net_structure += [32 for _ in range(2)]
net_structure += [0]
net_structure += [16 for _ in range(4)]
net_structure += [0]

def assister_loss(y_true, y_pred):
    return KB.in_train_phase(KL.categorical_crossentropy(y_true, y_pred),
                             KB.zeros_like(KL.categorical_crossentropy(y_true, y_pred)))

def ignore(y_true, y_pred):
    return KL.mean_absolute_error(y_pred, y_pred)

model_params = {"model":Charlie,
            "input_points":(64,2),
            "output_categories":(2,2),
            "weight_ratio":[0.01,1.],
            "node_per_layer":net_structure,
            "dropout_rate":.1,
            "max_norm":3.,
            "layer_activation":"relu",
            "output_activation":"softmax",
            "optimizer":"AdaDelta",
            "loss":"categorical_crossentropy",
            "a_loss":"categorical_crossentropy"
            }

n1 = 0
accuracy1 = (0.,0.)
while n1 < 5 and accuracy1[1] <= .75:
    print("Training on spirals, no holes")
    model1, accuracy1, epoch1, t1, keys1 = run(train_params, model_params, data_params, v_params, True)
    n1 += 1

n2 = 0
accuracy2 = (0.,0.)
while n2 < 5 and accuracy2[0] <= .75:
    data_params["data_count_h"] = 15000
    data_params["data_count_s"] = 15000
    data_params["data_count_c"] = 1
    data_params["data_count_a"] = 1
    #data_params["data_count_n"] = 0
    
    v_params["data_count_h"] = 5000
    v_params["data_count_s"] = 5000
    v_params["data_count_c"] = 1
    v_params["data_count_a"] = 1
    #v_params["data_count_n"] = 0
    
    model_params["weight_ratio"] = [1.,1.]
    model_params["loss"] = "categorical_crossentropy"
    model_params["a_loss"] = "categorical_crossentropy"
    
    train_params["epochs"] = 100
    train_params["patience"] = 10
    
    print("Training on spirals, with holes")
    model1 = Charlie_recompile(model1, model_params)
    model2, accuracy2, epoch2, t2, keys2 = run(train_params, model_params, data_params, v_params, True, model=model1)
    n2 += 1

#n3 = 0
#accuracy3  = (0.,0.)
#while n3 < 5 and accuracy3[0] <= .75:
#    data_params["data_count_h"] = 15000
#    data_params["data_count_s"] = 15000
#    data_params["data_count_c"] = 0
#    data_params["data_count_a"] = 0
#    data_params["data_count_n"] = 2
#    data_params["ignore_spiral_labels"] = True
#    
#    v_params["data_count_h"] = 5000
#    v_params["data_count_s"] = 5000
#    v_params["data_count_c"] = 0
#    v_params["data_count_a"] = 0
#    v_params["data_count_n"] = 2
#    v_params["ignore_spiral_labels"] = True
#    
#    model_params["weight_ratio"] = [1.,.01]
#    model_params["loss"] = "categorical_crossentropy"
#    model_params["a_loss"] = "categorical_crossentropy"
#    
#    train_params["epochs"] = 100
#    train_params["patience"] = 10
#    
#    print("Training on no spirals, only holes")
#    model2 = Charlie_recompile(model2, model_params)
#    model, accuracy, epoch3, t3, keys3 = run(train_params, model_params, data_params, v_params, True, model=model2)
#    n3 += 1

#t = t1 + t2 + t3

v_params = data_params.copy()
v_params["data_count_h"] = 1000
v_params["data_count_s"] = 1000
v_params["data_count_c"] = 0
v_params["data_count_a"] = 0
v_params["data_count_n"] = 1
v_params["ignore_spiral_labels"] = True
shapes, data, labels = data_gen(v_params)
