#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 20:03:38 2019

@author: william
"""
from generators import data_gen
import matplotlib.pyplot as plt
import numpy as np
from keras import backend
from keras.callbacks import callbacks as cb
from save import Save_Manager
import time
import gc
import os

def run(train_params, model_params, data_params, v_params=None, plot=False, path=os.path.join(os.getcwd(), "Saves")):
    backend.clear_session()
    gc.collect()
    print("Generating Model")
    data_manager = Save_Manager(path)
    model = model_params["model"](model_params)
    print(model.summary())
    
    print("Generating Data")
    ret = data_manager.load_data(data_params)
    if ret is not None:
        data , labels, key = ret
        shapes, _ = data_manager.load_shapes(data_params)
    else:
        shapes, data, labels = data_gen(data_params)
        key = data_manager.save_data(data, labels, data_params)
        data_manager.save_shapes(shapes, data_params)
    
    if train_params["patience"] >= 0:
        callbacks = [cb.EarlyStopping(monitor='val_loss', patience=train_params["patience"], restore_best_weights=train_params["restore_best_weights"])]
    else:
        callbacks = []
    
    if train_params["verification"]:
        ret = data_manager.load_data(v_params)
        if ret is not None:
            v_data , v_labels, _ = ret
            v_shapes, _ = data_manager.load_shapes(v_params)
        else:
            v_shapes, v_data, v_labels = data_gen(v_params)
            data_manager.save_data(v_data, v_labels, v_params)
            data_manager.save_shapes(v_shapes, v_params)
        print("Begining fit")
        start = time.time()
        history = model.fit(data, labels, epochs=train_params["epochs"], validation_data=(v_data,v_labels), shuffle=True, callbacks=callbacks)
    else:
        print("Begining fit")
        start = time.time()
        history = model.fit(data, labels, epochs=train_params["epochs"], shuffle=True, callbacks=callbacks)

    end = time.time()
    path = os.path.join(path, str(key)+"_models")
    model_manager = Save_Manager(path)
    model_manager.save_model(model, model_params, train_params)
    
    if plot:
        #Plot training & validation accuracy values
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
        print()
        
        fig, axs = plt.subplots(2,2)
        v_shapes[0].plot(axs[0,0])
        v_shapes[-1].plot(axs[0,1])
        n= data_params["data_points"]
        xs = [i*(data_params["upper"]-data_params["lower"])/n+(data_params["lower"]-data_params["upper"])/2 for i in range(n)]
        ax = axs[1,0]
        ax.plot(xs,v_data[0][0])
        ax = axs[1,1]
        ax.plot(xs,v_data[0][-1])
    
    backend.clear_session()
    gc.collect()
    epoch = np.argmin(history.history["val_loss"])
    accuracy = history.history["val_accuracy"][epoch]
    return model, accuracy, epoch, end-start

def train_default_params():
    return {"epochs":20, "verification":True, "patience":5, "restore_best_weights":True}

if __name__ == "__main__":
    from models import Adam_default_params
    from generators import data_default_params
    
    plt.close("all")
    train_params = train_default_params()
    model_params = Adam_default_params()
    data_params = data_default_params(600)
    v_params = data_default_params(200)
    
    model, accuracy, epoch, t = run(train_params, model_params, data_params, v_params, True)
    print(accuracy)
    print(epoch)
    print(t)
