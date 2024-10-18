
import tensorflow as tf
import numpy as np
from SRC.Loss1D import make_loss

dtype="float64"
tf.keras.backend.set_floatx(
    dtype
)


@tf.function
def update_u(u_model, u_bases, P, neurons_final, x0, v0, x, npts):
    reg = 10**(-11) 
    t_0 = tf.zeros((1,1), dtype = dtype)
    
    p1, p2 = tf.unstack(P,axis=-1)
    
    with tf.autodiff.ForwardAccumulator(x, tf.ones_like(x)) as t1:
        with tf.autodiff.ForwardAccumulator(x, tf.ones_like(x)) as t2:
            u_b = u_bases(x)
        u_t_b = t2.jvp(u_b)
    u_tt_b = t1.jvp(u_t_b)
    
    
    with tf.autodiff.ForwardAccumulator(t_0, tf.ones_like(t_0)) as t3:
        u0 = u_bases(t_0)
    u_t0 = t3.jvp(u0)
    
    B = (p1*u_tt_b + u_t_b + p2*u_b)/(np.sqrt(npts))
    
    B = tf.concat([u0, B], axis=0)
    B = tf.concat([u_t0, B], axis=0)
    
    
    L = tf.zeros((npts, 1), dtype = dtype)
    L = tf.concat([tf.constant([[x0]], dtype = dtype), L], axis=0)
    L = tf.concat([tf.constant([[v0]], dtype = dtype), L], axis=0)

    weights_optimal = (tf.reshape(tf.linalg.lstsq(B, L, l2_regularizer = reg),[neurons_final]))
    u_model.layers[-1].w.assign(weights_optimal)
    
    
    return reg
