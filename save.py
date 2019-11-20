#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:47:12 2019

@author: william
"""
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
                    self.dictionary = {"data":{}, "shapes":{}}
        else:
            self.dictionary = {"data":{}, "shapes":{}}

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
            key = len(self.dictionary["data"])+1
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
            key = len(self.dictionary["shapes"])+1
            self.dictionary["shapes"][key] = dictionary
            self.update_dict()
        with open(os.path.join(self.path, str(key)+"_shape.pkl"), "wb") as file:
            pickle.dump(shapes, file)
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
