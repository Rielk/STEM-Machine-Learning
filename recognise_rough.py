# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 00:32:55 2020

@author: William
"""
from models import Adam, Adam_default_params
from generators import data_default_params
import matplotlib.pyplot as plt
from runner import run
import numpy as np

plt.close("all")

#Make the model
model_params = Adam_default_params()
model_params["input_points"] = (32,3)
model_params["node_per_layer"] = [32,32,32,0,16,16]
model_params["layer_activation"] = "relu"

#Produce learning data
data_params = data_default_params(6000)
data_params["angles"] = (0,np.pi/4,np.pi/2)

v_params = data_default_params(2000)
v_params["angles"] = (0,np.pi/4,np.pi/2)

train_params = {"model":Adam, "epochs":2, "verification":True}

model = run(train_params, model_params, data_params, v_params, True)
