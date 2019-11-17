#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 20:03:38 2019

@author: william
"""
from generators import data_gen
import matplotlib.pyplot as plt
from models import Adam

def run(train_params, model_params, data_params, v_params=None, plot=False):
    model = train_params["model"](model_params)
    _, data, labels = data_gen(data_params)
    if train_params["verification"]:
        _, v_data, v_labels = data_gen(v_params)
        history = model.fit(data, labels, epochs=train_params["epochs"], validation_data=(v_data,v_labels), shuffle=True)
    else:
        history = model.fit(data, labels, epochs=train_params["epochs"], shuffle=True)
    
    if plot:
        # Plot training & validation accuracy values
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        
        # Plot training & validation loss values
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        
        print()
        print("Prediction with hole:\t",model.predict(v_data)[0])
        print("Prediction without hole:\t",model.predict(v_data)[-1])
        
        fig, axs = plt.subplots(2,2)
        _[0].plot(axs[0,0])
        _[1].plot(axs[1,0])
        _[-1].plot(axs[0,1])
        _[-2].plot(axs[1,1])
    
    return model, history

def train_default_params():
    return {"model":Adam, "epochs":20, "verification":True}

if __name__ == '__main__':
    from models import Adam_default_params
    from generators import data_default_params
    
    plt.close("all")
    train_params = train_default_params()
    model_params = Adam_default_params()
    data_params = data_default_params(600)
    v_params = data_default_params(200)
    
    run(train_params, model_params, data_params, v_params, True)
