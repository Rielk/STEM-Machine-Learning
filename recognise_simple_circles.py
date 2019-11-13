#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 19:59:44 2019

@author: william
"""
from keras.layers import Input, Dense
from keras.models import Model
from generators import data_gen
import matplotlib.pyplot as plt

plt.close("all")

#Make the model
inputs = Input(shape=(32,), name="inputs")
x = Dense(32, activation="relu", name="dense1")(inputs) #Dense1
x = Dense(32, activation="relu", name="dense2")(x) #Dense2
x = Dense(32, activation="relu", name="dense3")(x) #Dense3
x = Dense(16, activation="relu", name="dense4")(x) #Dense4
x = Dense(16, activation="relu", name="dense5")(x) #Dense5
outputs = Dense(2, activation="sigmoid", name="output")(x)
model = Model(inputs=inputs, outputs=outputs)

#Compile model
model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])

data_needed = False
#Produce learning data
params = {"data_count_h":30000,
          "data_count_s":30000,
        
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
if data_needed:
    _, data, labels = data_gen(params)
    data = data[:,0]
    #Clear the shapes from memory
    _ = None
    print("Finished the training data")

#Produce validation data
v_params = {"data_count_h":10000,
          "data_count_s":10000,
        
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

if data_needed:
    _, v_data, v_labels = data_gen(v_params)
    v_data = v_data[:,0]

#Then fit the data
history = model.fit(data, labels, epochs=20, validation_data=(v_data,v_labels), shuffle=True)

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
