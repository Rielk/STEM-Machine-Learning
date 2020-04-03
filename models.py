#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 11:31:39 2019

@author: william
"""
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.layers import concatenate, Conv1D, Reshape
from keras.layers import Flatten, MaxPool1D
from keras.models import Model 
from keras.constraints import maxnorm
from generators import data_gen, data_default_params
import keras.losses as KL
import numpy as np

def Adam(params=None):
    """
    Params is a dictionary with key corresponding to values:
    input_points                  -tuple      -first index is points in a projection, second is the number of projections
    node_per_layer                -iter       -A list of the number of nodes in each dense layer, 0 indicates the concatenation layer
    number_of_single_layers       -int        -Alternative definition:The number of layers before concatenation
    number_of_collection_layers   -int        -Alternative definition:The number of layers after concatenation
    dropout_rate                  -float      -The frequency for the dropout layers
    max_norm                      -float      -The Max Norm for the dense layers
    layer_activation              -string     -The type of actication to use in the layers
    output_activation             -string     -The type of activation to use in the final layer
    optimizer                     -string     -The optimizer to use for training
    loss                          -string     -The loss function to use when training
    """
    if params is None:
        return "Adam"
    else:
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
            merge_layer = 0
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
            for i in range(params["number_of_single_layers"]+params["number_of_collection_layers"]):
                dense_layers.append(Dense(params["input_points"][0], activation=activation, name="dense_{}".format(i+1), kernel_constraint=maxnorm(max_norm)))
                dense_layers.append(Dropout(drop_rate))
                dense_layers.append(BatchNormalization())
            merge_layer = 3*params["number_of_single_layers"]
        
        outs = []    
        for x in inputs:
            for layer in dense_layers[:merge_layer]:
                x = layer(x)
            outs.append(x)
                
        try:
            activation = params["output_activation"]
        except KeyError:
            pass
        if len(outs) > 1:
            x = concatenate(outs)
        else:
            x = outs[0]
        
        for layer in dense_layers[merge_layer:]:
            x = layer(x)
        
        outputs = Dense(2, activation=activation, name="output")(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        optimizer = params["optimizer"]
        loss = params["loss"]
        
        model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
        return model

def Brian(params=None):
    """
    Params is a dictionary with key corresponding to values:
    input_points                  -tuple      -first index is points in a projection, second is the number of projections
    node_per_layer                -iter       -A list of the number of nodes in each dense layer, 0 indicates the concatenation layer
    dropout_rate                  -float      -The frequency for the dropout layers
    max_norm                      -float      -The Max Norm for the dense layers
    layer_activation              -string     -The type of actication to use in the layers
    output_activation             -string     -The type of activation to use in the final layer
    optimizer                     -string     -The optimizer to use for training
    loss                          -string     -The loss function to use when training
    """
    if params is None:
        return "Brian"
    else:
        inputs = []
        for n in range(params["input_points"][1]):
            i = Input(shape=(params["input_points"][0],), name="input_{}".format(n+1))
            inputs.append(i)
        
        drop_rate = params["dropout_rate"]
        max_norm = params["max_norm"]
        activation = params["layer_activation"]
        
        layers = []
        mod = 1
        merge_layer = 0
        for i,node_count in enumerate(params["node_per_layer"]):
            if node_count == 0:
                merge_layer = 3*i
                mod = 0
                continue
            if type(node_count) is tuple:
                layers.append(Conv1D(node_count[0], node_count[1], activation=activation, name="conv1D_{}".format(i+mod), kernel_constraint=maxnorm(max_norm)))
            else:
                layers.append(Dense(node_count, activation=activation, name="dense_{}".format(i+mod), kernel_constraint=maxnorm(max_norm)))
            layers.append(Dropout(drop_rate))
            layers.append(BatchNormalization())
        
        r = Reshape((params["input_points"][0], 1))
        outs = []    
        for x in inputs:
            x = r(x)
            for layer in layers[:merge_layer]:
                x = layer(x)
            outs.append(x)
                
        try:
            activation = params["output_activation"]
        except KeyError:
            pass
        if len(outs) > 1:
            x = concatenate(outs, axis=-1)
        else:
            x = outs[0]
        
        for layer in layers[merge_layer:]:
            x = layer(x)
        
        x = Flatten()(x)
        
        outputs = Dense(params["output_categories"], activation=activation, name="output")(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        optimizer = params["optimizer"]
        loss = params["loss"]
        
        model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
        return model
   
def Brian_recompile(model, params):
    optimizer = params["optimizer"]
    loss = params["loss"]
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model
    
def Charlie(params=None):
    if params is None:
        return "Charlie"
    else:
        inputs = []
        for n in range(params["input_points"][1]):
            i = Input(shape=(params["input_points"][0],), name="input_{}".format(n+1))
            inputs.append(i)
        
        drop_rate = params["dropout_rate"]
        max_norm = params["max_norm"]
        activation = params["layer_activation"]
        
        layers = []
        mod = 1
        merge_layer = 0
        layer_tracker = 0
        for i,node_count in enumerate(params["node_per_layer"]):
            if node_count == 0:
                if merge_layer == 0:
                    merge_layer = layer_tracker
                    mod = 0
                    continue
                else:
                    layers.append(Flatten())
                    layer_tracker += 1
                    continue
            if type(node_count) is tuple:
                layers.append(Conv1D(node_count[0], node_count[1], activation=activation, name="conv1D_{}".format(i+mod), kernel_constraint=maxnorm(max_norm), padding="same"))
                layer_tracker += 1
            else:
                layers.append(Dense(node_count, activation=activation, name="dense_{}".format(i+mod), kernel_constraint=maxnorm(max_norm)))
                layer_tracker += 1
            layers.append(Dropout(drop_rate))
            layers.append(BatchNormalization())
            layer_tracker += 2
        
        r = Reshape((params["input_points"][0], 1))
        outs = []    
        for x in inputs:
            x = r(x)
            for layer in layers[:merge_layer]:
                x = layer(x)
            outs.append(x)
                
        try:
            activation = params["output_activation"]
        except KeyError:
            pass
        if len(outs) > 1:
            x = concatenate(outs, axis=-1)
        else:
            x = outs[0]
        
        for layer in layers[merge_layer:]:
            x = layer(x)
        
        output1 = Dense(params["output_categories"][0], activation=activation, name="output_main")(x)
        output2 = Dense(params["output_categories"][1], activation=activation, name="output_assister")(x)
        outputs = [output1, output2]
        model = Model(inputs=inputs, outputs=outputs)
        
        optimizer = params["optimizer"]
        loss = params["loss"]
        a_loss = params["a_loss"]
        loss_weights = params["weight_ratio"]
        model.compile(optimizer=optimizer,
                      loss={"output_main":loss,
                            "output_assister":a_loss},
                      loss_weights=loss_weights, metrics=["accuracy"])
        return model
    
def Charlie_recompile(model, params):
    optimizer = params["optimizer"]
    loss = params["loss"]
    a_loss = params["a_loss"]
    loss_weights = params["weight_ratio"]
    model.compile(optimizer=optimizer,
                  loss={"output_main":loss,
                        "output_assister":a_loss},
                  loss_weights=loss_weights, metrics=["accuracy"])
    return model    
    
def Dave(params=None):
    if params is None:
        return "Dave"
    else:
        inputs = []
        for n in range(params["input_points"][1]):
            i = Input(shape=(params["input_points"][0],), name="input_{}".format(n+1))
            inputs.append(i)
        
        drop_rate = params["dropout_rate"]
        max_norm = params["max_norm"]
        activation = params["layer_activation"]
        
        layers = []
        mod = 1
        layer_tracker = 0
        merge_layer = 0
        for i,node_count in enumerate(params["node_per_layer"]):
            if node_count == 0:
                if merge_layer == 0:
                    merge_layer = layer_tracker
                    mod = 0
                    continue
                else:
                    layers.append(Flatten())
                    layer_tracker += 1
                    continue
            if type(node_count) is tuple:
                layers.append(Conv1D(node_count[0], node_count[1], activation=activation, name="conv1D_{}".format(i+mod), kernel_constraint=maxnorm(max_norm), padding="same"))
                layers.append(MaxPool1D(node_count[2], name="max_pooling1D_{}".format(i+mod)))
                layer_tracker += 2
            else:
                layers.append(Dense(node_count, activation=activation, name="dense_{}".format(i+mod), kernel_constraint=maxnorm(max_norm)))
                layer_tracker += 1
            layers.append(Dropout(drop_rate))
            layers.append(BatchNormalization())
            layer_tracker += 2
        
        r = Reshape((params["input_points"][0], 1))
        outs = []    
        for x in inputs:
            x = r(x)
            for layer in layers[:merge_layer]:
                x = layer(x)
            outs.append(x)
                
        try:
            activation = params["output_activation"]
        except KeyError:
            pass
        if len(outs) > 1:
            x = concatenate(outs, axis=-1)
        else:
            x = outs[0]
        
        for layer in layers[merge_layer:]:
            x = layer(x)
        
        output1 = Dense(params["output_categories"][0], activation=activation, name="output_main")(x)
        output2 = Dense(params["output_categories"][1], activation=activation, name="output_assister")(x)
        output = concatenate([output1, output2])
        outputs = [output]
        model = Model(inputs=inputs, outputs=outputs)
        
        optimizer = params["optimizer"]
        loss = KL.get(params["loss"])
        a_loss = KL.get(params["a_loss"])
        loss_weights = params["weight_ratio"]
        weight_main = loss_weights[0]
        weight_assi = loss_weights[1]
        def losses(y_true, y_pred):
            true_main = y_true[:,:params["output_categories"][0]]
            true_assi = y_true[:,params["output_categories"][0]:]
            pred_main = y_pred[:,:params["output_categories"][0]]
            pred_assi = y_pred[:,params["output_categories"][0]:]
            loss_main = loss(true_main, pred_main)
            loss_assi = a_loss(true_assi, pred_assi)
            return weight_main*loss_main + weight_assi*loss_assi
#            return KL.mean_absolute_error(y_pred, y_pred)
        model.compile(optimizer=optimizer, loss=losses, metrics=["accuracy"])
        return model
    
def Dave_recompile(model, params):
    optimizer = params["optimizer"]
    loss = KL.get(params["loss"])
    a_loss = KL.get(params["a_loss"])
    loss_weights = params["weight_ratio"]
    weight_main = loss_weights[0]
    weight_assi = loss_weights[1]
    def losses(y_true, y_pred):
        true_main = y_true[:,:params["output_categories"][0]]
        true_assi = y_true[:,params["output_categories"][0]:]
        pred_main = y_pred[:,:params["output_categories"][0]]
        pred_assi = y_pred[:,params["output_categories"][0]:]
        loss_main = loss(true_main, pred_main)
        loss_assi = a_loss(true_assi, pred_assi)
        return weight_main*loss_main + weight_assi*loss_assi
#            return KL.mean_absolute_error(y_pred, y_pred)
    model.compile(optimizer=optimizer, loss=losses, metrics=["accuracy"])
    return model
    
def Eliot(params=None):
    if params is None:
        return "Eliot"
    else:
        inputs = []
        for n in range(params["input_points"][1]):
            i = Input(shape=(params["input_points"][0],), name="input_{}".format(n+1))
            inputs.append(i)
        
        drop_rate = params["dropout_rate"]
        max_norm = params["max_norm"]
        activation = params["layer_activation"]
        
        layers = []
        mod = 1
        merge_layer = 0
        layer_tracker = 0
        for i,node_count in enumerate(params["node_per_layer"]):
            if node_count == 0:
                if merge_layer == 0:
                    merge_layer = layer_tracker
                    mod = 0
                    continue
                else:
                    layers.append(Flatten())
                    layer_tracker += 1
                    continue
            if type(node_count) is tuple:
                layers.append(Conv1D(node_count[0], node_count[1], activation=activation, name="conv1D_{}".format(i+mod), kernel_constraint=maxnorm(max_norm)))
                layers.append(MaxPool1D(node_count[2], name="max_pooling1D_{}".format(i+mod)))
                layer_tracker += 2
            else:
                layers.append(Dense(node_count, activation=activation, name="dense_{}".format(i+mod), kernel_constraint=maxnorm(max_norm)))
                layer_tracker += 1
            layers.append(Dropout(drop_rate))
            layers.append(BatchNormalization())
            layer_tracker += 2
        
        r = Reshape((params["input_points"][0], 1))
        outs = []    
        for x in inputs:
            x = r(x)
            for layer in layers[:merge_layer]:
                x = layer(x)
            outs.append(x)
                
        try:
            activation = params["output_activation"]
        except KeyError:
            pass
        if len(outs) > 1:
            x = concatenate(outs, axis=-1)
        else:
            x = outs[0]
        
        for layer in layers[merge_layer:]:
            x = layer(x)
        
        outputs = Dense(params["output_categories"], activation=activation, name="output_main")(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        optimizer = params["optimizer"]
        loss = params["loss"]
        
        model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
        return model

def Frank(params=None):
    if params is None:
        return "Frank"
    else:
        inputs = []
        for n in range(params["input_points"][1]):
            i = Input(shape=(params["input_points"][0],), name="input_{}".format(n+1))
            inputs.append(i)
        
        drop_rate = params["dropout_rate"]
        max_norm = params["max_norm"]
        activation = params["layer_activation"]
        
        layers = []
        mod = 1
        merge_layer = 0
        layer_tracker = 0
        for i,node_count in enumerate(params["node_per_layer"]):
            if node_count == 0:
                if merge_layer == 0:
                    merge_layer = layer_tracker
                    mod = 0
                    continue
                else:
                    layers.append(Flatten())
                    layer_tracker += 1
                    continue
            if type(node_count) is tuple:
                layers.append(Conv1D(node_count[0], node_count[1], activation=activation, name="conv1D_{}".format(i+mod), kernel_constraint=maxnorm(max_norm), data_format="channels_first"))
                try:
                    node_count[2]
                    layers.append(MaxPool1D(node_count[2], name="max_pooling1D_{}".format(i+mod), data_format="channels_first"))    
                    layer_tracker += 2
                except IndexError:
                    layer_tracker += 1
            else:
                layers.append(Dense(node_count, activation=activation, name="dense_{}".format(i+mod), kernel_constraint=maxnorm(max_norm)))
                layer_tracker += 1
            layers.append(Dropout(drop_rate))
            layers.append(BatchNormalization())
            layer_tracker += 2
            
        r = Reshape((1, params["input_points"][0]))
        outs = []    
        for x in inputs:
            x = r(x)
            for layer in layers[:merge_layer]:
                x = layer(x)
            outs.append(x)
                
        try:
            activation = params["output_activation"]
        except KeyError:
            pass
        if len(outs) > 1:
            x = concatenate(outs, axis=1)
        else:
            x = outs[0]
        
        for layer in layers[merge_layer:]:
            x = layer(x)
        
        outputs = Dense(params["output_categories"], activation=activation, name="output")(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        optimizer = params["optimizer"]
        loss = params["loss"]
        
        model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
        return model
   
def Frank_recompile(model, params):
    optimizer = params["optimizer"]
    loss = params["loss"]
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model

def Grant(params=None):
    """
    Params is a dictionary with key corresponding to values:
    input_points                  -tuple      -first index is points in a projection, second is the number of projections
    node_per_layer                -iter       -A list of the number of nodes in each dense layer, 0 indicates the concatenation layer
    dropout_rate                  -float      -The frequency for the dropout layers
    max_norm                      -float      -The Max Norm for the dense layers
    layer_activation              -string     -The type of actication to use in the layers
    output_activation             -string     -The type of activation to use in the final layer
    optimizer                     -string     -The optimizer to use for training
    loss                          -string     -The loss function to use when training
    """
    if params is None:
        return "Grant"
    else:
        inputs = []
        for n in range(params["input_points"][1]):
            i = Input(shape=(params["input_points"][0],), name="input_{}".format(n+1))
            inputs.append(i)
        
        drop_rate = params["dropout_rate"]
        max_norm = params["max_norm"]
        activation = params["layer_activation"]
        
        layers = []
        mod = 1
        merge_layer = 0
        layer_tracker = 0
        for i,node_count in enumerate(params["node_per_layer"]):
            if node_count == 0:
                merge_layer = layer_tracker
                mod = 0
                continue
            if node_count == -1:
                layers.append(Reshape((params["input_points"][0], 1)))
                layer_tracker += 1
                continue
            if type(node_count) is tuple:
                layers.append(Conv1D(node_count[0], node_count[1], activation=activation, name="conv1D_{}".format(i+mod), kernel_constraint=maxnorm(max_norm), padding="same"))
                try:
                    node_count[2]
                    layers.append(MaxPool1D(node_count[2], name="max_pooling1D_{}".format(i+mod)))    
                    layer_tracker += 2
                except IndexError:
                    layer_tracker += 1
            else:
                layers.append(Dense(node_count, activation=activation, name="dense_{}".format(i+mod), kernel_constraint=maxnorm(max_norm)))
                layer_tracker += 1
            layers.append(Dropout(drop_rate))
            layers.append(BatchNormalization())
            layer_tracker += 2
        outs = []    
        for x in inputs:
            for layer in layers[:merge_layer]:
                x = layer(x)
            outs.append(x)
                
        try:
            activation = params["output_activation"]
        except KeyError:
            pass
        if len(outs) > 1:
            x = concatenate(outs, axis=-1)
        else:
            x = outs[0]
        
        for layer in layers[merge_layer:]:
            x = layer(x)
        
        x = Flatten()(x)
        
        outputs = Dense(params["output_categories"], activation=activation, name="output")(x)
        model = Model(inputs=inputs, outputs=outputs)
        
        optimizer = params["optimizer"]
        loss = params["loss"]
        
        model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
        return model

def Adam_default_params():
    return {"model":Adam,
            "input_points":(32,2),
            "node_per_layer":[32,32,32,0,16,16],
            "dropout_rate":.2,
            "max_norm":3.,
            "layer_activation":"relu",
            "output_activation":"softmax",
            "optimizer":"sgd",
            "loss":"categorical_crossentropy",
            }
    

if __name__ == '__main__':
    model_params = Adam_default_params()
#    del(model_params["node_per_layer"])
#    model_params["number_of_single_layers"] = 3
#    model_params["number_of_collection_layers"] = 2

    model = Adam(model_params)
    #Produce learning data
    data_params = data_default_params(600)
    data_params["angles"] = (0, np.pi/4)
    _, data, labels = data_gen(data_params)
    #Clear the shapes from memory
    _ = None
    
    #Produce validation data
    v_params = data_default_params(200)
    v_params["angles"] = (0, np.pi/4)
    v_params["background"] = 19.
    v_params["noise"] = .0011
    
    _, v_data, v_labels = data_gen(v_params)
    #Clear the shapes from memory
    _ = None
    
    #Then fit the data
    training_epochs = 10
    history = model.fit(data, labels, epochs=training_epochs, validation_data=(v_data,v_labels), shuffle=True)
