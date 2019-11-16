#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 11:31:39 2019

@author: william
"""
from keras.layers import Input, Dense, Dropout, BatchNormalization, concatenate
from keras.models import Model
from keras.constraints import maxnorm
from generators import data_gen, data_default_params
import numpy as np

def Adam(params):
    inputs = []
    for n in range(params["input_points"][1]):
        i = Input(shape=(params["input_points"][0],), name="input_{}".format(n+1))
        inputs.append(i)
        
    
    drop_rate = params["dropout_rate"]
    max_norm = params["max_norm"]
    activation = params["layer_activation"]
    
    dense_layers = []
    try:
        params["node_per_layer"]
        mod = 1
        for i,node_count in enumerate(params["node_per_layer"]):
            if node_count == 0:
                merge_layer = 3*i
                mod = 0
                continue
            dense_layers.append(Dense(node_count, activation=activation, name="dense_{}".format(i+mod), kernel_constraint=maxnorm(max_norm)))
            dense_layers.append(Dropout(drop_rate))
            dense_layers.append(BatchNormalization())
    except KeyError:
        params["number_of_single_layers"]
        for i,node_count in enumerate(params["number_of_single_layers"]+params["number_of collection_layers"]):
            dense_layers.append(Dense(params["input_points"], activation=activation, name="dense_{}".format(i+1), kernel_constraint=maxnorm(max_norm)))
            dense_layers.append(Dropout(drop_rate))
            dense_layers.append(BatchNormalization())
        params["number_of_single_layers"] = 3*i
    
    outs = []    
    for x in inputs:
        for layer in dense_layers[:merge_layer]:
            x = layer(x)
        outs.append(x)
            
    try:
        activation = params["output_activation"]
    except KeyError:
        pass
    x = concatenate(outs)
    
    for layer in dense_layers[merge_layer:]:
        x = layer(x)
    
    outputs = Dense(2, activation=activation, name="output")(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = params["optimizer"]
    loss = params["loss"]
    
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model
    
def Adam_default_params():
    return {"input_points":(32,2),
            "node_per_layer":[32,32,32,0,16,16],
            "dropout_rate":.2,
            "max_norm":3.,
            "layer_activation":"relu",
            "output_activation":"sigmoid",
            "optimizer":"sgd",
            "loss":"categorical_crossentropy",
            "training_epochs":10
            }
    

if __name__ == '__main__':
    model_params = Adam_default_params

    model = Adam(model_params)
    #Produce learning data
    data_params = data_default_params(600)
    data_params["angles"] = (0, np.pi/4)
    _, data, labels = data_gen(data_params)
    data = [data[:,0], data[:,1]]
    #Clear the shapes from memory
    _ = None
    
    #Produce validation data
    v_params = data_default_params(200)
    v_params["angles"] = (0, np.pi/4)
    v_params["background"] = 19.
    v_params["noise"] = .0011
    
    _, v_data, v_labels = data_gen(v_params)
    v_data = [v_data[:,0], v_data[:,1]]
    #Clear the shapes from memory
    _ = None
    
    #Then fit the data
    history = model.fit(data, labels, epochs=model_params["training_epochs"], validation_data=(v_data,v_labels), shuffle=True)
