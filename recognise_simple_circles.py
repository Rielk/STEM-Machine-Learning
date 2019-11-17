#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 19:59:44 2019

@author: william
"""
from models import Adam, Adam_default_params
from generators import data_gen
import matplotlib.pyplot as plt

plt.close("all")

#Make the model
model_params = Adam_default_params()
model_params["input_points"] = (32,1)

model = Adam(model_params)

data_needed = True
#Produce learning data
data_params = {"data_count_h":3000,
               "data_count_s":3000,
               
               "r":1.,    
               "sig_r":.1,
               "angle_count":90,
               "centre":(0,0),
               "sig_c":(.3,.3),
               "sig":.0,
               "h_range":.4,
               "h_r":.2,
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
if data_needed:
    _, data, labels = data_gen(data_params)
    #Clear the shapes from memory
    _ = None

v_data_needed = True
#Produce validation data
v_params = {"data_count_h":1000,
          "data_count_s":1000,
        
          "r":1.,    
          "sig_r":.1,
          "angle_count":90,
          "centre":(0,0),
          "sig_c":(.3,.3),
          "sig":.0,
          "h_range":.4,
          "h_r":.2,
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

if v_data_needed:
    _, v_data, v_labels = data_gen(v_params)

#Then fit the data
history = model.fit(data, labels, epochs=100, validation_data=(v_data,v_labels), shuffle=True)

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
