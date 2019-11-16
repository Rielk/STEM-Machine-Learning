#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 11:31:39 2019

@author: william
"""
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.constraints import maxnorm
from generators import data_gen

def Adam(params):
    #Make the model
    inputs = Input(shape=(params["input_points"],), name="inputs")
    
    drop_rate = params["dropout_rate"]
    max_norm = params["max_norm"]
    activation = params["layer_activation"]
    
    x = inputs
    try:
        params["node_per_layer"]
        for i,node_count in enumerate(params["node_per_layer"]):
            x = Dense(32, activation=activation, name="dense{}".format(i+1), kernel_constraint=maxnorm(max_norm))(x)
            x = Dropout(drop_rate)(x)
            x = BatchNormalization()(x)
    except KeyError:
        pass
    
    try:
        activation = params["output_activation"]
    except KeyError:
        pass
    outputs = Dense(2, activation=activation, name="output")(x)
    model = Model(inputs=inputs, outputs=outputs)
    
    optimizer = params["optimizer"]
    loss = params["loss"]
    
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model
    

if __name__ == '__main__':
    params = {"input_points":32,
              "node_per_layer":[32,32,32,16,16],
              "dropout_rate":.2,
              "max_norm":3.,
              "layer_activation":"relu",
              "output_activation":"sigmoid",
              "optimizer":"sgd",
              "loss":"categorical_crossentropy"
              }

    model = Adam(params)
    #Produce learning data
    data_params = {"data_count_h":300,
                   "data_count_s":300,
                   
                   "r":1.,    
                   "sig_r":.1,
                   "angle_count":90,
                   "centre":(0,0),
                   "sig_c":(.3,.3),
                   "sig":.0,
                   "h_range":.4,
                   "h_r":.4,
                   "sig_hr":.02,
                   
                   "data_points":32,
                   "angles":(0,),
                   "sig_as":(.02,),
                   "about":(0.,0.),
                   "lower":-1.5,
                   "upper":1.5,
                   "background":20.,
                   "gauss":.5,
                   "noise":.001,
                   }

    _, data, labels = data_gen(data_params)
    data = data[:,0]
    #Clear the shapes from memory
    _ = None
    
    #Produce validation data
    v_params = {"data_count_h":100,
              "data_count_s":100,
            
              "r":1.,    
              "sig_r":.1,
              "angle_count":90,
              "centre":(0,0),
              "sig_c":(.3,.3),
              "sig":.0,
              "h_range":.4,
              "h_r":.4,
              "sig_hr":.02,
            
              "data_points":32,
              "angles":(0,),
              "sig_as":(.04,),
              "about":(0.,0.),
              "lower":-1.5,
              "upper":1.5,
              "background":19.,
              "gauss":.5,
              "noise":.0011,
              }
    
    _, v_data, v_labels = data_gen(v_params)
    v_data = v_data[:,0]
    #Clear the shapes from memory
    _ = None
    
    #Then fit the data
    history = model.fit(data, labels, epochs=10, validation_data=(v_data,v_labels), shuffle=True)
