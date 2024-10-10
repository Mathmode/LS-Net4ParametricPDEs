#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited on Sept, 2024

@author: curiarteb
"""

from config import A,B,N,M1,M2,DTYPE
from enforce_dirichlet import enforce_dirichlet
import os, tensorflow as tf, numpy as np
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
keras.config.set_dtype_policy(DTYPE)
keras.mixed_precision.set_global_policy(DTYPE)

keras.utils.set_random_seed(1234)

@tf.function
def lelu(z):
    return tf.where(z > 0, z, 2 * z)

@tf.function
def singular_behavior(inputs):
    x, y = inputs  # Both x and y are of shape [batch_size, 1]
    
    ones = tf.repeat(tf.ones_like(x), repeats=N//5+N%5, axis=1)    
    
    # Compute (x - c1)^2 + (y - c2)^2 for all centers at once using broadcasting
    r_squared1 = (x - 1/2)**2 + (y - 1/2)**2
    r_squared2 = (x + 1/2)**2 + (y - 1/2)**2
    r_squared3 = (x - 1/2)**2 + (y + 1/2)**2
    r_squared4 = (x + 1/2)**2 + (y + 1/2)**2
    
    # Apply lelu function to the results for all centers
    
    #tf.print("Size r_squared1", r_squared1)
    mask1 = tf.repeat(lelu(1/4 - r_squared1), repeats=N//5, axis=1)
    mask2 = tf.repeat(lelu(1/4 - r_squared2), repeats=N//5, axis=1) 
    mask3 = tf.repeat(lelu(1/4 - r_squared3), repeats=N//5, axis=1) 
    mask4 = tf.repeat(lelu(1/4 - r_squared4), repeats=N//5, axis=1)
    
    return tf.concat([ones,mask1,mask2,mask3,mask4],axis=1)

class u_net(keras.Model):
    def __init__(self, **kwargs):
        super(u_net, self).__init__()
        
        # Hidden layers
        self.hidden_layers = [keras.layers.Dense(units=N, activation="sigmoid", use_bias=True) for i in range(3)]
        
        #Last (linear) layer
        self.linear_layer = keras.layers.Dense(units=1, activation=None, use_bias=False)
        
        # To enforce Dirichlet boundary conditions and regularity-conforming conditions
        self.enforce_dirichlet = enforce_dirichlet()
        self.singular_behavior = singular_behavior
        
    def build(self, input_shape):
        super(u_net, self).build(input_shape)

    # Returns the vector output of the neural network
    def call_vect(self, inputs):
        out = tf.concat(inputs, axis=1)
        for layer in self.hidden_layers:
            out = layer(out)
            
        # To impose boundary conditions
        #out = out*(inputs[0]-A)*(inputs[1]-A)*(inputs[0]-B)*(inputs[1]-B)
        out = out*self.enforce_dirichlet(inputs)
        out = out*self.singular_behavior(inputs)
        
        return out
    
    # Returns the scalar output of the neural network
    def call(self, inputs):
        out = self.call_vect(inputs)
        return self.linear_layer(out)
    
    # Computes the derivative with respect to the input via forward autodiff
    def dfwd(self, inputs):
        x,y = inputs
        with tf.autodiff.ForwardAccumulator(primals=x, tangents=tf.ones_like(x)) as tapeX, tf.autodiff.ForwardAccumulator(primals=y, tangents=tf.ones_like(y)) as tapeY:
            u = self([x,y])
        duX = tapeX.jvp(u)
        duY = tapeY.jvp(u)
        return [duX,duY]
    
    # Computes the derivative with respect to the input via backward autodiff
    def dbwd(self, inputs):
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(inputs)
            u = self(inputs)
        du = tape.gradient(u,inputs)
        return du
    
    # Computes the derivative with respect to the input via forward autodiff
    # for the vector output
    def dfwd_vect(self, inputs):
        x,y = inputs
        with tf.autodiff.ForwardAccumulator(primals=x, tangents=tf.ones_like(x)) as tapeX, tf.autodiff.ForwardAccumulator(primals=y, tangents=tf.ones_like(y)) as tapeY:
            u = self.call_vect([x,y])
        duX = tapeX.jvp(u)
        duY = tapeY.jvp(u)
        return [duX, duY]
    
    # Computes the 2nd order derivative with respect to the input 
    # via backward autodiff for the vector output
    def ddbwd(self, inputs):
        x,y = inputs
        with tf.GradientTape(watch_accessed_variables=False) as touter:
            touter.watch(inputs)
            with tf.GradientTape(watch_accessed_variables=False) as tinner:
                tinner.watch(inputs)
                u = self(inputs)
            du = tinner.gradient(u,inputs)
        ddu = touter.gradient(du,inputs)
        return ddu

    
class test_functions(keras.Model):
    def __init__(self, spectrum_size=[M1,M2], **kwargs):
        super(test_functions,self).__init__()
        
        self.M1,self.M2 = spectrum_size
        self.eval = lambda x,y: tf.concat([2/(np.pi*np.sqrt(m1**2+m2**2))*tf.sin(m1*np.pi*(x-A)/(B-A))*tf.sin(m2*np.pi*(y-A)/(B-A)) for m1 in range(1, self.M1+1) for m2 in range(1, self.M2+1)], axis=1) # sines normalized in H10
        self.devalX = lambda x,y: tf.concat([2*m1/(np.pi*np.sqrt(m1**2+m2**2))*tf.cos(m1*np.pi*(x-A)/(B-A))*tf.sin(m2*np.pi*(y-A)/(B-A)) for m1 in range(1, self.M1+1) for m2 in range(1, self.M2+1)], axis=1) # dsinesX normalized  in H10
        self.devalY = lambda x,y: tf.concat([2*m2/(np.pi*np.sqrt(m1**2+m2**2))*tf.sin(m1*np.pi*(x-A)/(B-A))*tf.cos(m2*np.pi*(y-A)/(B-A)) for m1 in range(1, self.M1+1) for m2 in range(1, self.M2+1)], axis=1) # dsinesY normalized  in H10
        self.ddevalXX = lambda x,y: tf.concat([-2*m1**2/(np.pi*np.sqrt(m1**2+m2**2))*tf.sin(m1*np.pi*(x-A)/(B-A))*tf.sin(m2*np.pi*(y-A)/(B-A)) for m1 in range(1, self.M1+1) for m2 in range(1, self.M2+1)], axis=1) # dsinesY normalized  in H10
        self.ddevalYY = lambda x,y: tf.concat([-2*m2**2/(np.pi*np.sqrt(m1**2+m2**2))*tf.sin(m1*np.pi*(x-A)/(B-A))*tf.sin(m2*np.pi*(y-A)/(B-A)) for m1 in range(1, self.M1+1) for m2 in range(1, self.M2+1)], axis=1) # dsinesY normalized  in H10
        
    # Evaluation of the test functions
    def call(self, inputs):
        x,y=inputs
        return self.eval(x,y)
    
    # Evaluation of the spatial gradient of the test functions
    def gradient(self, inputs):
        x,y=inputs
        return [self.devalX(x,y), self.devalY(x,y)]
    
    # Evaluation of the spatial laplacian of the test functions
    def laplacian(self, inputs):
        x,y=inputs
        return self.ddevalXX(x,y) + self.ddevalYY(x,y)