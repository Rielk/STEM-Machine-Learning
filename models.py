#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 11:31:39 2019

@author: william
"""
from keras.layers import Input, Dense, Dropout, BatchNormalization, Merge
from keras.models import Model
from keras.constraints import maxnorm
from generators import data_gen, default_params
import numpy as np

def Adam(params):
    inputs_list = []
    for n in range(params["input_points"][1]):
        inputs = Input(shape=(params["input_points"][0],), name="input{}".format(n))
        inputs_list.append(inputs)
        
    
    drop_rate = params["dropout_rate"]
    max_norm = params["max_norm"]
    activation = params["layer_activation"]
    
    for x in inputs_list:
        try:
            params["node_per_layer"]
            for i,node_count in enumerate(params["node_per_layer"]):
                x = Dense(32, activation=activation, name="dense{}".format(i+1), kernel_constraint=maxnorm(max_norm))(x)
                x = Dropout(drop_rate)(x)
                x = BatchNormalization()(x)
        except KeyError:
            params["number_of_layers"]
            for i,node_count in enumerate(params["number_of_layer"]):
                x = Dense(params["input_points"], activation=activation, name="dense{}".format(i+1), kernel_constraint=maxnorm(max_norm))(x)
                x = Dropout(drop_rate)(x)
                x = BatchNormalization()(x)
            
    try:
        activation = params["output_activation"]
    except KeyError:
        pass
    x = Merge(inputs_list)
    outputs = Dense(2, activation=activation, name="output")(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = params["optimizer"]
    loss = params["loss"]
    
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model
    

if __name__ == '__main__':
    model_params = {"input_points":(32,2),
              "node_per_layer":[32,32,32,16,16],
              "dropout_rate":.2,
              "max_norm":3.,
              "layer_activation":"relu",
              "output_activation":"sigmoid",
              "optimizer":"sgd",
              "loss":"categorical_crossentropy",
              "training_epochs":10
              }

    model = Adam(model_params)
    #Produce learning data
    data_params = default_params(600)
    data_params["angles"] = (0, np.pi/4)
    _, data, labels = data_gen(data_params)
    data = np.array([data[:,0], data[:,1]])
    #Clear the shapes from memory
    _ = None
    
    #Produce validation data
    v_params = default_params(200)
    v_params["angles"] = (0, np.pi/4)
    v_params["background"] = 19.
    v_params["noise"] = .0011
    
    _, v_data, v_labels = data_gen(v_params)
    v_data = np.array([v_data[:,0], v_data[:,1]])
    #Clear the shapes from memory
    _ = None
    
    #Then fit the data
    history = model.fit(data, labels, epochs=model_params["training_epochs"], validation_data=(v_data,v_labels), shuffle=True)
