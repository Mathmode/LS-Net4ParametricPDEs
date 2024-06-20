
from SRC.ProblemHelmholtz1D import rhs, u_exact
import numpy as np
import tensorflow as tf
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
dtype='float64' 
keras.mixed_precision.set_dtype_policy(dtype) 


dtype="float64"
tf.keras.backend.set_floatx(
    dtype
)

###############################################################################
###############################################################################
###############################################################################
def indicator(x,a,b):
    div1 = (1+tf.math.sign(x-a))/2
    div2 = (1+(tf.math.sign(b-x)))/2
    return div1*div2

def sigma(x,a,b,c):
    return a*indicator(x,0,1./3.)+b*indicator(x,1./3.,2./3.)+c*indicator(x,2/3.,1.)
###############################################################################
###############################################################################
###############################################################################+
@tf.function
def make_loss(u_model, xi, x, DST, DCT, norms): 
    
    a,b,c,w = tf.unstack(xi,axis=-1)
    
    with tf.autodiff.ForwardAccumulator(x, tf.ones_like(x)) as t2:
        u = u_model(x)
    du = t2.jvp(u)
    
    high = tf.einsum("iI,i->I",DCT,sigma(x,a,b,c)*du)
    low = tf.einsum("iI,i->I",DST,rhs(x)-w**2*u)
    
    FFT = high + low
    return tf.reduce_sum((FFT)**2)

###############################################################################
###############################################################################
###############################################################################