
import tensorflow as tf
import numpy as np

dtype="float64"
tf.keras.backend.set_floatx(
    dtype
)


# calculating collocation points
def collocation_points(a, b, npts): 
    
    a = tf.cast(a, tf.float64)
    b = tf.cast(b, tf.float64)
    h = tf.linspace(a, b, npts)
    
    diff = tf.abs(h[1:] - h[:-1])
    x = tf.linspace(a + diff[0]/2, b, npts)
    x = tf.reshape(x, (npts, 1))
    return x