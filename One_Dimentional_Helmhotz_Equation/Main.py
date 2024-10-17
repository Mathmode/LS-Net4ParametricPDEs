# -*- coding: utf-8 -*-
"""
Created on 18 July 2024

@author: Shima Baharlouei
"""

import tensorflow as tf
import numpy as np
import scipy.io

from SRC.Architecture1D import make_u_model
from SRC.Test_functions import test_functions
from SRC.Training import train_step, lr_schedule, loss_evaluation
from SRC.data import u_exact, g_1, g_2, indicator, sigma, f

# Set the default dtype to float64
dtype = "float64"
tf.keras.backend.set_floatx(dtype)

###############################################################################
###############################################################################
###############################################################################
# Data configuration

neurons = 5  # Number of neurons in hidden layers
neurons_final = 30  # Number of neurons in the last layer

npts = 1000  # Number of integration points
eps_npts = 2  # Additional integration points for evaluating validation loss
nmodes = 400  # Number of modes
# Parameter ranges for random sampling
max_p1 = 2  # Max value of parameter p1 (in log)
min_p1 = -np.log10(2) - 1  # Min value of parameter p1 (in log)
max_p2 = 2  # Max value of parameter p2 (in log)
min_p2 = -np.log10(2) - 1  # Min value of parameter p2 (in log)
max_p3 = 10  # Max value of parameter p3
min_p3 = 0  # Min value of parameter p3

dat_len = 1000  # Length of the dataset
dat_dim = 3  # Dimensionality of the data
batch_size = 1  # Batch size for training

num_epochs = 5000  # Number of epochs
initial_learning_rate = 10 ** (-1)  # Initial learning rate
final_learning_rate = 10 ** (-4)  # Final learning rate

# Save initial data configuration for later use
initial_data = {
    'npts': npts,
    'eps_npts': eps_npts,
    'max_p1': max_p1,
    'min_p1': min_p1,
    'max_p2': max_p2,
    'min_p2': min_p2,
    'max_p3': max_p3,
    'min_p3': min_p3,
    'dat_len': dat_dim,
    'neurons_final': neurons_final,
    'nmodes': nmodes
}
scipy.io.savemat('Saved_Models/initial_data.mat', initial_data)

# Generate test functions
x, DST, DCT, DCT_a, DCT_b, norms = test_functions(nmodes, npts)
x_val, DST_val, DCT_val, DCT_a_val, DCT_b_val, norms_val = test_functions(nmodes, npts + eps_npts)

###############################################################################
###############################################################################
###############################################################################
# Define validation parameter set for each epoch

# Build the model
u_model, u_bases = make_u_model(neurons, neurons_final=neurons_final, activation='sigmoid')
u_model.summary()
u_bases.summary()

# Set up the optimizer with initial learning rate
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
u_model.compile(optimizer=optimizer)

# Save the model before training
u_model.save('Saved_Models/u_model_before_training.keras')
u_bases.save('Saved_Models/u_bases_before_training.keras')

###############################################################################
# Training loop
###############################################################################

Loss = []  # List to store training losses
Loss_val = []  # List to store validation losses

# Define random validation parameter set
P_val_1 = 10 ** (tf.random.uniform([dat_len, 1], dtype=dtype, minval=min_p1, maxval=max_p1))
P_val_2 = 10 ** (tf.random.uniform([dat_len, 1], dtype=dtype, minval=min_p2, maxval=max_p2))
P_val_3 = tf.random.uniform([dat_len, 1], dtype=dtype, minval=min_p3, maxval=max_p3)
P_val = tf.stack([P_val_1, P_val_2, P_val_3], axis=-1)

# Loop over each epoch
for epoch in range(num_epochs):
    # Define random training parameter set for the current epoch
    P_data_1 = 10 ** (tf.random.uniform([dat_len, 1], dtype=dtype, minval=min_p1, maxval=max_p1))
    P_data_2 = 10 ** (tf.random.uniform([dat_len, 1], dtype=dtype, minval=min_p2, maxval=max_p2))
    P_data_3 = tf.random.uniform([dat_len, 1], dtype=dtype, minval=min_p3, maxval=max_p3)
    P_data = tf.stack([P_data_1, P_data_2, P_data_3], axis=-1)

    # Adjust the learning rate according to the schedule
    new_learning_rate = lr_schedule(epoch, initial_learning_rate, final_learning_rate, num_epochs)
    optimizer.learning_rate.assign(new_learning_rate)

    # Perform one training step and evaluate training loss
    Loss += [train_step(u_model, u_bases, P_data, x, DST, DCT, DCT_a, DCT_b, norms, neurons_final, optimizer)]

    # Evaluate validation loss
    Loss_val += [loss_evaluation(u_model, u_bases, P_val, x_val, DST_val, DCT_val, DCT_a_val, DCT_b_val, norms_val, neurons_final, optimizer)]

    # Print progress after each epoch
    print('========================================')
    print('============= Epoch:', epoch, '/', num_epochs, '=============')
    print('Loss_training:', float(Loss[-1]))
    print('Loss_validation:', float(Loss_val[-1]))

# Save the model after training
u_model.save('Saved_Models/u_model_after_training.keras')
u_bases.save('Saved_Models/u_bases_after_training.keras')

# Save the training history
mat_dict_history = {
    'iterations': num_epochs,
    'history': Loss,
    'history_val': Loss_val,
}
scipy.io.savemat('Data_history_1.mat', mat_dict_history)

