
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
    a=tf.constant([0.],dtype=dtype)
    b=tf.constant([1.],dtype=dtype)
    x = tf.constant([(i + 0.5 )/npts for i in range(npts)],dtype=dtype)
    
    DCT = tf.stack([np.sqrt(1/((np.pi*(k))**2 + 1))*np.sqrt(2/np.pi)*tf.math.cos((k)*x*np.pi) for k in range(1,nmodes - 1)],axis=-1)/npts
    DST = -tf.stack([np.sqrt(1/((np.pi*(k))**2 + 1))*np.sqrt(2/np.pi)*(np.pi*(k))*tf.math.sin((k)*x*np.pi) for k in range(1,nmodes - 1)],axis=-1)/npts
    DCT_b = tf.stack([np.sqrt(1/((np.pi*(k))**2 + 1))*np.sqrt(2/np.pi)*tf.math.cos((k)*b*np.pi) for k in range(1,nmodes - 1)],axis=-1)
    DCT_a = tf.stack([np.sqrt(1/((np.pi*(k))**2 + 1))*np.sqrt(2/np.pi)*tf.math.cos((k)*a*np.pi) for k in range(1,nmodes - 1)],axis=-1)
    norms = tf.constant([[ 1/((np.pi*(1-k))**2 + 1)] for k in range(1,nmodes)],dtype=dtype)
    
    
    phi_0 =tf.constant([[np.sqrt(2/np.pi) - 1/np.sqrt(np.pi)] for i in range(npts)],dtype=dtype)/npts
    d_phi_0 = tf.constant([[0.] for i in range(npts)],dtype=dtype)/npts
    DCT = tf.concat([phi_0, DCT], axis=1)
    DST = tf.concat([d_phi_0, DST], axis=1)
    
    phi_0_a = tf.constant([[np.sqrt(2/np.pi) - 1/np.sqrt(np.pi)]], dtype=dtype)
    phi_0_b = phi_0_a
    DCT_b = tf.concat([phi_0_b, DCT_b], axis=1)
    DCT_a = tf.concat([phi_0_a, DCT_a], axis=1)
    
    return x, DST, DCT, DCT_a, DCT_b, norms