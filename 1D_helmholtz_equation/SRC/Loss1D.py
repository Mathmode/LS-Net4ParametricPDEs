
from SRC.data import u_exact, g_1, g_2, indicator, sigma, f
from SRC.My_Mul import my_multiplication
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
###############################################################################+
@tf.function
def make_loss(u_model, p, x, DST, DCT, DCT_a, DCT_b, norms): 
    
    a = tf.constant([0.],dtype=dtype)
    b = tf.constant([1.],dtype=dtype)
    s1, s2, K = tf.unstack(p,axis=-1)

    with tf.autodiff.ForwardAccumulator(x, tf.ones_like(x)) as t1:
         u = u_model(x)
    du = t1.jvp(u)

    high =  tf.transpose(DST) @ (tf.stack([sigma(x, s1, s2)],axis=-1)*du)
    low = tf.transpose(DCT) @ (-K**2*u + f(x, s1, s2, K))

    u_b = u_model(b)
    u_a = u_model(a)

    b_1 = g_2() + my_multiplication(np.array([[0, 1]]), tf.sqrt(s2)*K*u_b)
    a_1 = g_1() + my_multiplication(np.array([[0, 1]]), tf.sqrt(s1)*K*u_a)
    B_b = tf.transpose(DCT_b) @ b_1
    B_a = tf.transpose(DCT_a) @ a_1

    FFT = high + low - B_b - B_a

    loss = tf.reduce_sum((FFT)**2)
    

    return loss

###############################################################################
###############################################################################
###############################################################################