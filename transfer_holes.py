# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 00:32:55 2020

@author: William
"""
from recognise_spirals import data_params, v_params, model_params, train_params
from recognise_spirals import model, accuracy, epoch, t, n, keys
from generators import data_gen
from models import Brian_recompile
import keras.models as KM
import matplotlib.pyplot as plt
from runner import run
plt.close("all")

#Produce learning data
data_params["h_range"] = .4
data_params["h_r"] = .05
data_params["sig_hr"] = .005
data_params["data_count_h"] = 30000
data_params["data_count_s"] = 30000
#data_params["dummy_out"] = True
del data_params["data_count_c"]
del data_params["data_count_a"]
del data_params["s_r"]

v_params["h_range"] = .4
v_params["h_r"] = .1
v_params["sig_hr"] = .01
v_params["data_count_h"] = 10000
v_params["data_count_s"] = 10000
#v_params["dummy_out"] = True
del v_params["data_count_c"]
del v_params["data_count_a"]
del v_params["s_r"]

train_params["epochs"] = 100
train_params["patience"] = 10
train_params["notes"] = "This was a transfer from recognise_spirals"

for n2 in range(3):
    model2 = KM.clone_model(model)
    model2 = Brian_recompile(model2, model_params)
    print("Training on shapes with holes")
    model2, accuracy2, epoch2, t2, keys2 = run(train_params, model_params, data_params, v_params, True, model=model2)
    if accuracy2 >= .75:
        v_params["data_count_h"] = 1000
        v_params["data_count_s"] = 1000
        shapes, data, labels = data_gen(v_params)
        if model2.evaluate(data, labels)[1] >= model2.evaluate([data[0], data[0]], labels)[1] + .1:
            break
else:
    raise KeyboardInterrupt("Too many attempts")

T = t + t2
print("Spiral Accuracy = ", accuracy)
print("Spiral Epoch =", epoch)
print("Spiral attempt ", n)
print("Spiral keys: ", keys)
print()
print("Hole Accuracy = ", accuracy2)
print("Hole Epoch =", epoch2)
print("Hole attempt ", n2)
print("Hole keys: ", keys2)
print()
print("t = ",T)
