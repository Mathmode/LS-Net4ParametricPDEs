EXPERIMENT_REFERENCE = "helmholtz2D" # Reference
RESULTS_FOLDER = "results" # Folder name for saving results
#
import tensorflow as tf, numpy as np # Necessary packages (if needed)
#
DTYPE = "float32" # float type
SOURCE = lambda x,y : tf.zeros_like(x)
LIFT = lambda x,y : tf.math.cos(np.pi/2*x)
DLIFTX = lambda x,y : -np.pi/2*tf.math.sin(np.pi/2*x)
A = tf.constant(-1., dtype=DTYPE) # Left end point of the domain of integration
B = tf.constant(1., dtype=DTYPE) # Right end point of the domain of integration
Aparam = tf.constant(1., dtype=DTYPE) # Left end point of the parameter domain 
Bparam = tf.constant(10., dtype=DTYPE) # Right end point of the parameter domain 
P = 32 # Parameter batch size
N = 75 # number of trial spanning functions (integer)
M1 = 75 # number of test basis functions
M2 = 75 # number of test basis functions (M1 x M2 must be an integer larger than N)
MVAL1 = 100 # larger than M1
MVAL2 = 100 # larger than M2
K1 = 4*M1 # number of integration points (integer larger than M1)
K2 = 4*M2 # number of integration points (integer larger than M2)
KVAL1 = 4*MVAL1 # number of integration points (integer larger than M1)
KVAL2 = 4*MVAL2 # number of integration points (integer larger than M2)
LEARNING_RATE = 0.01
EPOCHS = 10000 # Number of iterations
XYPLOT = [tf.expand_dims(tf.linspace(A, B, KVAL1), axis=1), tf.expand_dims(tf.linspace(A, B, KVAL2), axis=1)] # Domain sample for plotting
PI = tf.constant(3.14159265358979323846, dtype=DTYPE)
