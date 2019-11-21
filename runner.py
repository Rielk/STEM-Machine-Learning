#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 20:03:38 2019

@author: william
"""
from generators import data_gen
import matplotlib.pyplot as plt
from keras import backend
from save import Save_Manager
import gc
import os

def run(train_params, model_params, data_params, v_params=None, plot=False, path=os.path.join(os.getcwd(), "Saves")):
    data_manager = Save_Manager(path)
    model = model_params["model"](model_params)
    
    ret = data_manager.load_data(data_params)
    if ret is not None:
        data , labels, key = ret
    else:
        shapes, data, labels = data_gen(data_params)
        key = data_manager.save_data(data, labels, data_params)
        data_manager.save_shapes(shapes, data_params)
    
    if train_params["verification"]:
        ret = data_manager.load_data(v_params)
        if ret is not None:
            v_data , v_labels, _ = ret
        else:
            v_shapes, v_data, v_labels = data_gen(v_params)
            data_manager.save_data(v_data, v_labels, v_params)
            data_manager.save_shapes(v_shapes, v_params)
        history = model.fit(data, labels, epochs=train_params["epochs"], validation_data=(v_data,v_labels), shuffle=True)
    else:
        history = model.fit(data, labels, epochs=train_params["epochs"], shuffle=True)
    
    path = os.path.join(path, str(key)+"_models")
    model_manager = Save_Manager(path)
    model_manager.save_model(model, model_params, train_params)
    
    if plot:
        # Plot training & validation accuracy values
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("Model accuracy")
        plt.ylabel("Accuracy")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Test"], loc="upper left")
        
        # Plot training & validation loss values
        plt.figure()
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Model loss")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Test"], loc="upper left")
        plt.show()
        
        print()
        print("Prediction with hole:\t",model.predict(v_data)[0])
        print("Prediction without hole:\t",model.predict(v_data)[-1])
        
        fig, axs = plt.subplots(2,2)
        shapes[0].plot(axs[0,0])
        v_shapes[0].plot(axs[1,0])
        shapes[-1].plot(axs[0,1])
        v_shapes[-1].plot(axs[1,1])
    
    del model
    del history
    backend.clear_session()
    gc.collect()

def train_default_params():
    return {"epochs":20, "verification":True}

if __name__ == "__main__":
    from models import Adam_default_params
    from generators import data_default_params
    
    plt.close("all")
    train_params = train_default_params()
    model_params = Adam_default_params()
    data_params = data_default_params(600)
    v_params = data_default_params(200)
    
    run(train_params, model_params, data_params, v_params, False)
