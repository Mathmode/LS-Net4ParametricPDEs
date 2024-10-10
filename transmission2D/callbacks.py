#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited on Sept., 2024

@author: curiarteb
"""

import os, numpy as np
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from config import EPOCHS


class Validation(keras.callbacks.Callback):
    def __init__(self, sigma_val, num_of_validations=100):
        super().__init__()
        
        self.sigma_val = sigma_val
        self.validation_iterations = list(np.round(10**np.linspace(np.log10(1), np.log10(EPOCHS), num_of_validations)).astype(int))
        self.validation_iterations.append(EPOCHS-1)
        self.val_iteration = []
        self.val_int = []
        self.val_test = []
        
    def on_epoch_begin(self, epoch, logs=None):
        
        if epoch in self.validation_iterations:
            
            self.val_iteration.append(epoch)
            self.val_int.append(float(self.model.loss_intval.MC(self.sigma_val).numpy()))
            self.val_test.append(float(self.model.loss_testval.MC(self.sigma_val).numpy()))
            
    def on_train_end(self, logs=None):
        # Access the history attribute of the model
        history = self.model.history.history
        history["iteration_val"] = self.val_iteration
        history["loss_intval"] = self.val_int
        history["loss_testval"] = self.val_test
