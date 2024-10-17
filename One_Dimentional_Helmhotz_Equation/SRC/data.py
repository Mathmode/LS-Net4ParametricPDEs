# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 10:37:46 2023

@author: jamie.taylor
"""

import tensorflow as tf
import numpy as np

dtype="float64"
tf.keras.backend.set_floatx(
    dtype
)

sin=tf.math.sin
cos=tf.math.cos
sqrt=tf.math.sqrt
exp=tf.math.exp
i = tf.constant([0.0 + 1.0j], dtype=tf.complex64)


def indicator(x,a,b):
    div1 = (1+tf.math.sign(x-a))/2
    div2 = (1+(tf.math.sign(b-x)))/2
    return div1*div2

def sigma(x, s1, s2):
    return s1*indicator(x,0,1./2.) + \
           s2*indicator(x,1./2.,1.) 

def u_exact(x, s1, s2, K, dtype=tf.float64):
    
    i = tf.constant(0.0 + 1.0j, dtype=tf.complex64)
    s1 = tf.cast(s1, tf.complex64)
    s2 = tf.cast(s2, tf.complex64)
    K = tf.cast(K, tf.complex64)
    x = tf.cast(x, tf.complex64)
    
    a = ((-exp(0.5*i*K*(sqrt(s1) + sqrt(s2))/(sqrt(s1)*sqrt(s2))) +\
          exp(0.5*i*K/sqrt(s1)))*s2*sqrt(s1))/\
        (K**2*(sqrt(s2)*s1 + sqrt(s1)*s2))
    b = (-2*s1*sqrt(s2)*exp(-0.5*i*K/sqrt(s2)) - sqrt(s1)*s2 + sqrt(s2)*s1)/\
        (2*K**2*(sqrt(s2)*s1 + sqrt(s1)*s2))
    c = -(exp(i*K/(sqrt(s2))))/(2*K**2)
    
    u = a*exp(-i*K*x/sqrt(s1))*indicator(x,0,1./2.) + \
        (b*exp(i*K*x/sqrt(s2)) + c*exp(-i*K*x/sqrt(s2)) + 1/(K**2))*indicator(x,1./2.,1.) 
    
    return tf.cast(tf.stack([tf.math.real(u), tf.math.imag(u)], axis=-1), dtype=tf.float64)

def f(x, s1, s2, K, dtype=tf.float64):
    #aa = tf.constant([0.0 + 1.0j], dtype=tf.complex64)
    a = indicator(x,1./2., 1)
    
    return tf.cast(tf.stack([tf.math.real(a), tf.math.imag(a)], axis=-1), dtype=tf.float64)




def g_1(dtype=tf.float64):
   A = 0
   return tf.cast(tf.stack([[tf.math.real(A)], [tf.math.imag(A)]], axis=-1), dtype=tf.float64)

def g_2(dtype=tf.float64):                                                                             
   A = 0
   return tf.cast(tf.stack([[tf.math.real(A)], [tf.math.imag(A)]], axis=-1), dtype=tf.float64)


