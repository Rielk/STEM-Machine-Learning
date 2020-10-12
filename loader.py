# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 03:59:56 2020

@author: William
"""
from save import Save_Manager
import matplotlib.pyplot as plt
plt.close("all")

data_saves = Save_Manager(r"E:\Users\William\Documents - no sync\Project\Saves")
#model_saves = Save_Manager(r"E:\Users\William\Documents - no sync\Project\Incorrect Normalisation\0_models")
train_saves = Save_Manager(r"E:\Users\William\Documents - no sync\Project\Saves\0_models\5_model")
data, labels, _ = data_saves.load_data(data_saves.dictionary["data"][0])
shapes, _ = data_saves.load_shapes(data_saves.dictionary["shapes"][0])
model, history, _ = train_saves.load_trained(train_saves.dictionary["train"][0], n=3)

plt.figure()
try:
    plt.plot(history["accuracy"])
    plt.plot(history["val_accuracy"])
    acc1 = True
except KeyError:
    plt.plot(history["output_main_accuracy"])
    plt.plot(history["output_assister_accuracy"])
    plt.plot(history["val_output_main_accuracy"])
    plt.plot(history["val_output_assister_accuracy"])
    acc1 = False
    
plt.title("Network accuracy")
plt.ylabel("Accuracy")
#plt.yscale("log")
plt.xlabel("Epoch")
if acc1:
    plt.legend(["Train", "Validation"], loc="upper left")
else:
    plt.legend(["Train main", "Train assisst", "Test main", "Test assist"], loc="upper left")
plt.xlim(0, len(history["accuracy"])-1)
#plt.ylim(0.4,1)
plt.tight_layout()

# Plot training & validation loss values
plt.figure()
plt.plot(history["loss"])
plt.plot(history["val_loss"])
plt.title("Network loss")
plt.ylabel("Loss")
#plt.yscale("log")
plt.xlabel("Epoch")
plt.legend(["Train", "Validation"], loc="upper left")
#plt.ylim(0,1)
plt.xlim(0, len(history["accuracy"])-1)
plt.tight_layout()
plt.show()
            
fig, axs = plt.subplots(2,2)
i = 0
j = -1
lower = -1.5
upper = 1.5

shapes[i].rotate_coords()
shapes[j].rotate_coords()
shapes[i].plot(axs[0,0])
shapes[j].plot(axs[0,1])
axs[0,0].set_xlim(lower, upper)
axs[0,1].set_xlim(lower, upper)

n = len(data[0][i])
xs = [i*(upper-lower)/n+(lower-upper)/2 for i in range(n)]
ax = axs[1,0]
ax.plot(xs,data[0][i])
ax.set_xlim(lower,upper)
ax.set_ylim(0,1)
ax.set_aspect('equal', 'box')
ax.set_ylabel("Intensity")
ax.set_xlabel("Position")
ax = axs[1,1]
ax.plot(xs,data[0][j])
ax.set_xlim(lower,upper)
ax.set_ylim(0,1)
ax.set_aspect('equal', 'box')
ax.set_ylabel("Intensity")
ax.set_xlabel("Position")
plt.tight_layout()
     