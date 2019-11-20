#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:47:12 2019

@author: william
"""
from keras.models import load_model
import pickle
import os

class Save_Manager():
    def __init__(self, path):
        self.path = path
        if not os.path.exists(path):
            os.mkdir(path)
        dic_path = os.path.join(path, "dict.pkl")
        if os.path.exists(dic_path):
            with open(dic_path, "rb") as file:
                try:
                    self.dictionary = pickle.load(file)
                except EOFError:
                    self.dictionary = {"data":{}, "shapes":{},
                                       "model":{}, "train":{}}
        else:
            self.dictionary = {"data":{}, "shapes":{},
                               "model":{}, "train":{}}

    def update_dict(self):
        dic_path = os.path.join(self.path, "dict.pkl")
        with open(dic_path, "wb") as file:
            pickle.dump(self.dictionary, file)

    def save_data(self, data, labels, dictionary):
        for k in self.dictionary["data"]:
            if dictionary == self.dictionary["data"][k]:
                key = k
                break
        else:
            key = len(self.dictionary["data"])
            self.dictionary["data"][key] = dictionary
            self.update_dict()
        with open(os.path.join(self.path, str(key)+"_data.pkl"), "wb") as file:
            pickle.dump((data, labels), file)
        return key
            
    def save_shapes(self, shapes, dictionary):
        for k in self.dictionary["shapes"]:
            if dictionary == self.dictionary["shapes"][k]:
                key = k
                break
        else:
            key = len(self.dictionary["shapes"])
            self.dictionary["shapes"][key] = dictionary
            self.update_dict()
        with open(os.path.join(self.path, str(key)+"_shape.pkl"), "wb") as file:
            pickle.dump(shapes, file)
        return key
    
    def save_model(self, model, model_dict, train_dict):
        model_dict["model"] = model_dict["model"]()
        for k in self.dictionary["model"]:
            if model_dict == self.dictionary["model"][k]:
                key1 = k
                break
        else:
            key1 = len(self.dictionary["model"])
            self.dictionary["model"][key1] = model_dict
            self.update_dict()
        path = os.path.join(self.path, str(key1)+"_model")
        if not os.path.exists(path):
            os.mkdir(path)
        train_manager = Save_Manager(path)
        key2 = train_manager.save_trained(model, train_dict)
        return key1, key2
    
    def save_trained(self, model, dictionary):
        for k in self.dictionary["train"]:
            if dictionary == self.dictionary["train"][k]:
                key = k
                break
        else:
            key = len(self.dictionary["train"])
            self.dictionary["train"][key] = dictionary
            self.update_dict()
        path = os.path.join(self.path, str(key)+"_trained")
        if not os.path.exists(path):
            os.mkdir(path)
        path = os.path.join(path, str(len(os.listdir(path)))+".h5")
        model.save(path)
        return key
        
    def load_data(self, dictionary):
        for k in self.dictionary["data"]:
            if dictionary == self.dictionary["data"][k]:
                key = k
                break
        else:
            return None
        with open(os.path.join(self.path, str(key)+"_data.pkl"), "rb") as file:
            data, labels = pickle.load(file)
        return data, labels, key
    
    def load_model(self, model_dict, train_dict, n=0):
        model_dict["model"] = model_dict["model"]()
        for k in self.dictionary["model"]:
            if model_dict == self.model_dict["model"][k]:
                key1 = k
                break
        else:
            return None
        
        path = os.path.join(self.path, str(key1)+"_model")
        train_manager = Save_Manager(path)
        model, key2 = train_manager.load_trained(train_dict, n)
        return model, key1, key2
    
    def load_trained(self, dictionary, n=0):
        for k in self.dictionary["train"]:
            if dictionary == self.dictionary["train"][k]:
                key = k
                break
        else:
            return None
        path = os.path.join(self.path, str(key))
        path = os.path.join(path, str(n)+".h5")
        model = load_model(path)
        return model, key
