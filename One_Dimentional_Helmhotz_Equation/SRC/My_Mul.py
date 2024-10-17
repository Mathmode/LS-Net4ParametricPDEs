#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 13:01:17 2024

@author: sbahalouei
"""

import tensorflow as tf
import numpy as np

dtype="float64"
tf.keras.backend.set_floatx(
    dtype
)

def my_multiplication(a, b):
    c = a[:, 0]*b[:, 0] - a[:, 1]*b[:, 1]
    d = a[:, 0]*b[:, 1] + a[:, 1]*b[:, 0]
    return tf.stack([c, d], axis=-1)