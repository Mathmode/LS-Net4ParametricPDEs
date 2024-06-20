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


def test_functions(nmodes, npts):
    a=0
    b=1
    diff = (b-a)/npts
    x = tf.constant([a + (i + 0.5 )*diff for i in range(npts)],dtype=dtype)
    DST = tf.stack([np.sqrt(((np.pi*k)**2/(b-a)**2 + 1)**(-1))*np.sqrt(2/np.pi)*tf.math.sin(k*np.pi*(x - a)/(b-a)) 
                         for k in range(1,nmodes)],axis=-1)*diff
    DCT = tf.stack([np.sqrt(((np.pi*k)**2/(b-a)**2 + 1)**(-1))*np.sqrt(2/np.pi)*(np.pi*k/(b-a))*tf.math.cos(k*np.pi*(x - a)/(b-a)) 
                         for k in range(1,nmodes)],axis=-1)*diff
    norms = tf.constant([ np.sqrt(((np.pi*k)**2/(b-a)**2 + 1)**(-1)) for k in range(1,nmodes)],dtype=dtype)
    
    
    return x, DST, DCT, norms