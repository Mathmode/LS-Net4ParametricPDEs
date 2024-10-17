# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:37:46 2023

@author: jamie.taylor
"""

import tensorflow as tf
import numpy as np


sin=tf.math.sin
cos=tf.math.cos
sqrt=tf.math.sqrt
exp=tf.math.exp


def u_exact(x, p1, p2, x0, v0):
    if 1 - 4*p1*p2 >0:
        L1 = (-1 + tf.sqrt(1 - 4*p1*p2))/(2*p1)
        L2 = (-1 - tf.sqrt(1 - 4*p1*p2))/(2*p1)
        C2 = (v0 - L1*x0)/(-L1 + L2)
        C1 = x0 - C2
        out = C1*tf.exp(L1*x) + C2*tf.exp(L2*x)
    elif 1 - 4*p1*p2 == 0:
        L1 = -1 /(2*p1)
        L2 = L1
        C2 = v0 - L1*x0
        C1 = x0
        out = C1*tf.exp(L1*x) + C2*x*tf.exp(L2*x)
    else:
        alpha = -1 /(2*p1)
        beta = tf.sqrt(4*p1*p2 - 1)/(2*p1)
        C1 = x0
        C2 = (v0 - alpha*C1)/beta
        out = tf.exp(alpha*x)*(C1*tf.cos(beta*x) + C2*tf.sin(beta*x))
    return out