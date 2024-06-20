
import tensorflow as tf
from SRC.Loss1D import indicator, sigma
from SRC.ProblemHelmholtz1D import u_exact, rhs
from SRC.Loss1D import make_loss, indicator, sigma

dtype="float64"
tf.keras.backend.set_floatx(
    dtype
)


@tf.function
def update_u(u_model, u_bases, xi, nvar, x, DST, DCT, norms):

    reg = 10**(-10) 
    a,b,c,w = tf.unstack(xi,axis=-1)
    
    with tf.autodiff.ForwardAccumulator(x, tf.ones_like(x)) as t5:
        u_b = u_bases(x)
    du_b = t5.jvp(u_b)
    
    high = tf.einsum("i,iI->iI",sigma(x,a,b,c), du_b)
    high_bases = tf.einsum("iI,iJ->IJ",DCT,high)
    low_bases = tf.einsum("iI,iJ->IJ",DST,-w**2*u_b)
    B = (high_bases + low_bases)
    L = tf.expand_dims(tf.einsum("iI,i->I",DST, -rhs(x)), axis=1)
    
    weights_optimal = (tf.reshape(tf.linalg.lstsq(B, L, l2_regularizer = reg),[nvar]))
    u_model.layers[-1].w.assign(weights_optimal)
    
    return reg 
###############################################################################
###############################################################################
###############################################################################
