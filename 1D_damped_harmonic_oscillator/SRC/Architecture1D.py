
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


class MyLinearLayer(keras.layers.Layer):
    def __init__(self, n, dtype=tf.float64, **kwargs):
        super(MyLinearLayer, self).__init__(**kwargs)
        self.n = n
        self.dtype_ = dtype

        # Initialize the variable with a fixed random uniform distribution
        self.w = tf.Variable(tf.random.uniform([n], dtype=dtype) / (n**0.5), trainable=False, dtype=dtype)

    def call(self, inputs):
        px, L = inputs
        proto = tf.einsum("j,ij->i", self.w, L)
        return proto

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],)

    def get_config(self):
        config = super(MyLinearLayer, self).get_config()
        config.update({
            "n": self.n,
            "dtype": self.dtype_.name  # Convert dtype to string
        })
        return config
    

def make_u_model(neurons, activation=tf.math.sigmoid, neurons_final=False, dtype=tf.float64):
    kernel_regularizer = keras.regularizers.L2(l2=0)
    ker_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.3)
    b_init = keras.initializers.RandomNormal(mean=0.0, stddev=00.2)

    if not neurons_final:
        nn = neurons
    else:
        nn = neurons_final

    xvals = keras.layers.Input(shape=(1,), name="x_input", dtype=dtype)

    l1 = keras.layers.Dense(neurons, activation=activation, dtype=dtype,
                            kernel_regularizer=kernel_regularizer,
                            kernel_initializer=ker_init,
                            bias_initializer=b_init)(xvals - 0.5)
    
    for i in range(1):
        l1 = keras.layers.Dense(neurons, activation=activation, dtype=dtype, name=f"l{i+2}",
                                kernel_regularizer=kernel_regularizer,
                                kernel_initializer=ker_init,
                                bias_initializer=b_init)(l1)
                                
    l1 = keras.layers.Dense(nn, activation=activation, dtype=dtype, name="l12",
                            kernel_regularizer=kernel_regularizer,
                            kernel_initializer=ker_init,
                            bias_initializer=b_init)(l1)
     
    output = MyLinearLayer(nn, dtype=dtype)([xvals, l1])
    
    u_model = keras.Model(inputs=xvals, outputs=output)
    
    
    u_bases = keras.Model(inputs=xvals, outputs=l1)
    
    return u_model, u_bases



