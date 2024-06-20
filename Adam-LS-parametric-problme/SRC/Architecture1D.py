
import os, tensorflow as tf
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import numpy as np
dtype='float64' 
keras.mixed_precision.set_dtype_policy(dtype)

keras.utils.set_random_seed(1234)


dtype="float64"
tf.keras.backend.set_floatx(
    dtype
)




class MyLinearLayerBCConfMore(keras.layers.Layer):
    def __init__(self, n, dtype=tf.float64):
        super(MyLinearLayerBCConfMore, self).__init__()
        self.n = int(n / 3)
        self.w = self.add_weight(shape=(n,),
                                 initializer=tf.keras.initializers.RandomUniform(-1.0/(n**0.5), 1.0/(n**0.5)),
                                 trainable=False,
                                 dtype=dtype)

    def call(self, inputs):
        px, L = inputs
        proto = tf.einsum("j,ij->i", self.w, L)
        return proto

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],)

class BC_shima(tf.keras.layers.Layer):
    def __init__(self,n):
        super(BC_shima,self).__init__()
        self.n = int(n / 3)
    def call(self,inputs):
        px,L = inputs
        L = L*px*(1-px) #Boundary condition
        p1 = L[:, :self.n]
        p2 = L[:, self.n:2*self.n]* (1 + px + 0.5 * tf.math.abs(px - 1. / 3.))
        p3 = L[:, 2*self.n:3*self.n]* (1 + px + 0.5 * tf.math.abs(px - 2. / 3.))
        
        L = tf.concat([p1, p2, p3], axis=1)
        
        return L

def make_u_model(neurons, activation="tanh", neurons_final=False, dtype=tf.float64):
    kernel_regularizer = keras.regularizers.L2(l2=0)
    ker_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.3)
    b_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.2)

    if not neurons_final:
        nn = neurons
    else:
        nn = neurons_final

    xvals = keras.layers.Input(shape=(1,), name="x_input", dtype=dtype)

    l1 = keras.layers.Dense(neurons, activation=activation, dtype=dtype,
                            kernel_regularizer=kernel_regularizer,
                            kernel_initializer=ker_init,
                            bias_initializer=b_init)(xvals - 0.5)
    
    for i in range(3):
        l1 = keras.layers.Dense(neurons, activation=activation, dtype=dtype, name=f"l{i+2}",
                                kernel_regularizer=kernel_regularizer,
                                kernel_initializer=ker_init,
                                bias_initializer=b_init)(l1)
                                
    l1 = keras.layers.Dense(nn, activation=activation, dtype=dtype, name="l12",
                            kernel_regularizer=kernel_regularizer,
                            kernel_initializer=ker_init,
                            bias_initializer=b_init)(l1)
                   
    l1 = BC_shima(nn)([xvals,l1])         
    output = MyLinearLayerBCConfMore(nn, dtype=dtype)([xvals, l1])
    
    u_model = keras.Model(inputs=xvals, outputs=output)
    
    
    u_bases = keras.Model(inputs=xvals, outputs=l1)
    
    return u_model, u_bases



