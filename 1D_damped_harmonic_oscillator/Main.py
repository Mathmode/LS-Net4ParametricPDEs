# -*- coding: utf-8 -*-
"""
Created on 14 Aug 2024
@author: Shima Baharlouei
"""

############################################################
######################### Problem #########################
############################################################
# This code is going to solve the following damped problem:
#               p1u_tt + u_t + p2u = 0,
# where p1 and p2 are parameters.
# The initial conditions are u(0) = xo and u_t(0) = v0. 
# This code is used the LS-Net method with PINN 
############################################################
############################################################
############################################################


import tensorflow as tf
import time as time
import scipy.io

from SRC.Architecture1D import make_u_model
from SRC.Loss1D import make_loss
from SRC.LSArchitecture import update_u
from SRC.Training import train_step, lr_schedule, loss_evaluation
from SRC.Collocation_points import collocation_points

tf.random.set_seed(1234)

dtype="float64"
tf.keras.backend.set_floatx(
    dtype
)

############################################################
#### Initial data ##########################################
############################################################

neurons = 5 # Number of neurons in last hidden layers
neurons_final = 40 # Number of neurons in last layer

# Time interval domain (a, b]
a = 0 
b = 10

# Initial conditions
x0 = 0 # u(0) = x0
v0 = -50 # v(0) = v0

npts = 1000 # Number of used collocation points in training
eps_npts = 1 # I will use (npts + eps_npts) number of collocation points for evaluating loss on validation
eps_p = 0
# Parameters are choosen from interval (min_p, max_p)
max_p1 = 1.5 + eps_p # max value of parameters
min_p1 = -1.5 - eps_p # min value of parameters
max_p2 = 1.5 + eps_p # max value of parameters
min_p2 = -1.5  - eps_p # min value of parameters

dat_dim = 2 # Number of parameters, i.e. p1 and p2
dat_len =  500 # The dimention of training data for each parameter

batch_size = 1

num_epochs = 10000 # Number of epochs
initial_learning_rate = 0.1
final_learning_rate = 0.001

### Saving data in a file for plot ###
initial_data = {
    'a': a,
    'b': b,
    'x0': x0,
    'v0': v0,
    'npts': npts,
    'eps_npts': eps_npts,
    'max_p1': max_p1,
    'min_p1': min_p1, 
    'max_p2': max_p2,
    'min_p2': min_p2, 
    'dat_len': dat_dim,
    'neurons_final':neurons_final
}

scipy.io.savemat('Saved_Models/initial_data.mat', initial_data)


###############################################################################
###############################################################################
###############################################################################
# constructing the collocation points for training
x = collocation_points(a, b, npts)
# constructing the collocation points for evaluating loss on the validation
x_val = collocation_points(a, b, npts + eps_npts)

# validation parameter set define randomly in each epoch
P_val_1 = 10**(tf.random.uniform([dat_len,1],dtype=dtype,minval=min_p1,maxval=max_p1))
P_val_2 = (tf.random.uniform([dat_len,1],dtype=dtype,minval=10**min_p2,maxval=10**max_p2))
P_val = tf.stack([P_val_1, P_val_2], axis=-1)

u_model, u_bases = make_u_model(neurons,neurons_final=neurons_final,activation=tf.math.sigmoid)
u_model.summary()
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
u_model.compile(optimizer=optimizer,loss = make_loss)

# Save model before training
u_model.save('Saved_Models/u_model_before_training.keras')
u_bases.save('Saved_Models/u_bases_before_training.keras')

# Start the timer
start_time = time.time()

# Training loop
Loss = []
Loss_val = []
k = 0

for epoch in range(num_epochs):
    # training parameter set define randomly in each epoch
    P_data_1 = 10**(tf.random.uniform([dat_len,1],dtype=dtype,minval=min_p1,maxval=max_p1))
    P_data_2 = (tf.random.uniform([dat_len,1],dtype=dtype,minval=10**min_p2,maxval=10**max_p2))
    P_data = tf.stack([P_data_1, P_data_2], axis=-1)
    #Changing the learning rate
    new_learning_rate = lr_schedule(epoch, initial_learning_rate, final_learning_rate, num_epochs)
    optimizer.learning_rate.assign(new_learning_rate)
    Loss += [train_step(u_model, u_bases, P_data, neurons_final, optimizer, x0, v0, x, npts)]
    Loss_val += [loss_evaluation(u_model, u_bases, P_val, neurons_final, x0, v0, x_val, npts + eps_npts)]
    k += 1
    print('===========================================')
    print('=============', 'Epoch:', epoch, '/', num_epochs,'=============')
    print('Loss_training:', float(Loss[-1]))
    print('Loss_validation:', float(Loss_val[-1]))

# End the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

# Save model aftre training
u_model.save('Saved_Models/u_model_after_training.keras')
u_bases.save('Saved_Models/u_bases_after_training.keras')

mat_dict_history = {
    'iterations': num_epochs,
    'history': Loss,
    'history_val': Loss_val,
}

scipy.io.savemat('Data_history_1.mat', mat_dict_history)
###############################################################################
###############################################################################
###############################################################################
###############################################################################################################
      







