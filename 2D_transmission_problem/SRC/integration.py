#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited on Sept, 2024

@author: curiarteb
"""

from SRC.config import A, B, K1, K2, DTYPE
import tensorflow as tf

class integration_points_and_weights():
    def __init__(self, threshold=[K1, K2]):
        
        num_subintervalsX = threshold[0] // 2 + 1
        num_subintervalsY = threshold[1] // 2 + 1
        
        self.gridX = tf.linspace(A, B, num_subintervalsX)
        self.gridY = tf.linspace(A, B, num_subintervalsY)
            
        # Remaining elements    
        b_gridX = tf.expand_dims(self.gridX[1:], axis=1)
        a_gridX = tf.expand_dims(self.gridX[:-1], axis=1)
        b_gridY = tf.expand_dims(self.gridY[1:], axis=1)
        a_gridY = tf.expand_dims(self.gridY[:-1], axis=1)
        
        self.scaleX = (b_gridX - a_gridX) / 2
        self.scaleY = (b_gridY - a_gridY) / 2
        self.biasX = (b_gridX + a_gridX) / 2
        self.biasY = (b_gridY + a_gridY) / 2
        
    # Generates random integration points and weights for exact integration in p=1
    def generate_raw(self):
        
        ## Comment and uncommet to set fixed or stochastic quadrature points
        #x1 = tf.expand_dims(tf.random.uniform(shape=(len(self.gridX) - 1,), minval=-1, maxval=1, dtype=DTYPE), axis=1)
        x1 = tf.expand_dims(0.5*tf.ones(shape=[len(self.gridX) - 1], dtype=DTYPE), axis=1)
        x2 = -x1
        #y1 = tf.expand_dims(tf.random.uniform(shape=(len(self.gridY) - 1,), minval=-1, maxval=1, dtype=DTYPE), axis=1)
        y1 = tf.expand_dims(0.5*tf.ones(shape=[len(self.gridX) - 1], dtype=DTYPE), axis=1)
        y2 = -y1
        
        x_points = tf.concat([x1, x2], axis=1)
        y_points = tf.concat([y1, y2], axis=1)
        
        wx_points = tf.ones_like(x_points)
        wy_points = tf.ones_like(y_points)
        
        return [x_points, y_points, wx_points, wy_points]
    
    # Every time we call it, we produce new integration points and weights
    def __call__(self):
        x, y, wx, wy = self.generate_raw()
        
        x = self.scaleX * x + self.biasX
        y = self.scaleY * y + self.biasY
        wx = self.scaleX * wx
        wy = self.scaleY * wy
        
        x_con = tf.expand_dims(tf.concat(tf.unstack(x), axis=0), axis=1)
        y_con = tf.expand_dims(tf.concat(tf.unstack(y), axis=0), axis=1)
        wx_con = tf.expand_dims(tf.concat(tf.unstack(wx), axis=0), axis=1)
        wy_con = tf.expand_dims(tf.concat(tf.unstack(wy), axis=0), axis=1)
        
        x_rep = tf.repeat(x_con, y_con.shape[0], axis=0)
        y_rep = tf.tile(y_con, [x_con.shape[0], 1])
        
        wx_rep = tf.repeat(wx_con, wy_con.shape[0], axis=0)
        wy_rep = tf.tile(wy_con, [wx_con.shape[0], 1])
        
        w = wx_rep * wy_rep

        return [x_rep, y_rep, w]
    
    