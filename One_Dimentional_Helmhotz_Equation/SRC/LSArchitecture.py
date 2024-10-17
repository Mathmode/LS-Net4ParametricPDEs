
import tensorflow as tf
import numpy as np
from SRC.Loss1D import indicator, sigma
from SRC.Loss1D import make_loss, indicator, sigma
from SRC.data import u_exact, g_1, g_2, indicator, sigma, f
from SRC.My_Mul import my_multiplication

dtype="float64"
tf.keras.backend.set_floatx(
    dtype
)


@tf.function
def update_u(u_model, u_bases, p, nvar, x, DST, DCT, DCT_a, DCT_b, norms):

    
    a=tf.constant([0],dtype=dtype)
    b=tf.constant([1],dtype=dtype)
    reg = 10**(-10) 
    s1, s2, K = tf.unstack(p,axis=-1)
    
    #########################
    
    with tf.autodiff.ForwardAccumulator(x, tf.ones_like(x)) as t2:
        u = u_bases(x)
    du = t2.jvp(u)

    #########################
    
    high_bases = tf.einsum("iI,iJ->IJ",DST,tf.stack([sigma(x, s1, s2)],axis=-1)*du)
    low_bases = tf.einsum("iI,iJ->IJ",DCT, -K**2*u)
    
    ##### boundary parts of right hand side #####
    
    u_b = tf.einsum("iI,jJ->IJ",DCT_b, u_bases(b))
    u_a = tf.einsum("iI,jJ->IJ",DCT_a, u_bases(a))
    
    coef_b = np.array([[0, 1]])*tf.sqrt(s2)*K
    coef_a = np.array([[0, 1]])*tf.sqrt(s1)*K
    
    Boundary_real = u_b*coef_b[:, 0] + u_a*coef_a[:, 0]
    Boundary_imag = u_b*coef_b[:, 1] + u_a*coef_a[:, 1]
    
    #########################
    B_real = high_bases + low_bases - Boundary_real
    B_imag =  -Boundary_imag
    #########################
    ##### boundary parts of left hand side is zero #####
    
    ##### source terms (real and imaginary) #####
    Source = tf.einsum("iI,iJ->IJ",DCT, -f(x, s1, s2, K))
    Source_real = tf.expand_dims(Source[:, 0], axis=1)
    Source_imag = tf.expand_dims(Source[:, 1], axis=1)
    
    ##### Constructing matrix B and vector Source for LS system #####
    # Create the top and bottom parts of the block matrix
    top = tf.concat([B_real, -B_imag], axis=1)  # Concatenate A and B horizontally
    bottom = tf.concat([B_imag, B_real], axis=1)  # Concatenate B and A horizontally

    # Create the final block matrix by concatenating top and bottom vertically
    B = tf.concat([top, bottom], axis=0)
    Source = tf.concat([Source_real, Source_imag], axis=0)
    
    ######## LS solver #########
    weights_optimal = (tf.reshape(tf.linalg.lstsq(B, Source, l2_regularizer = reg),[2*nvar]))
    u_model.layers[-1].w[0:nvar].assign(weights_optimal[0:nvar])
    u_model.layers[-1].w[nvar:2*nvar].assign(weights_optimal[nvar:2*nvar])
    
    #loss = tf.reduce_sum((B*u_model.layers[-1].w -Source)**2)
    
    return reg
###############################################################################
###############################################################################
###############################################################################