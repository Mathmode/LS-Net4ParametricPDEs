
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

@tf.function
def make_loss(u_model, P, x0, v0, x): 
    
    p1, p2 = tf.unstack(P,axis=-1)
    t_0 = tf.zeros((1,1), dtype = dtype)

    with tf.GradientTape() as tape2:
        tape2.watch(x)
        with tf.GradientTape() as tape1:
            tape1.watch(x)
            u = u_model(x)
        u_t = tape1.gradient(u, x)
    u_tt = tape2.gradient(u_t, x)

    u_tt = tf.reshape(u_tt, (-1, 1))
    u_t = tf.reshape(u_t, (-1, 1))
    u = tf.reshape(u, (-1, 1))
    
    with tf.GradientTape() as tape3:
        tape3.watch(t_0)
        u0 = u_model(t_0)
    u_t0 = tape3.gradient(u0, t_0)
        
    # PINN method for the damped equation
    ode_loss = p1*u_tt + u_t + p2*u 
    
    # Initial conditions
    IC_loss_1 = u0 - tf.constant([[x0]], dtype = dtype)
    IC_loss_2 = u_t0 - tf.constant([[v0]], dtype = dtype)

    square_loss = tf.square(ode_loss)
    total_loss = tf.reduce_mean(square_loss) + tf.squeeze(tf.square(IC_loss_1) + tf.square(IC_loss_2))
 
    return total_loss

###############################################################################
###############################################################################
############################################################################### self.u_model = u_model
    
    