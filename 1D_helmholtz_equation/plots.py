#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 21:47:19 2024

@author: sbahalouei
"""

# Import required libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io  # For loading and saving .mat files
import keras
from tensorflow.keras.utils import get_custom_objects  # For registering custom layers

# Import custom functions and models
from SRC.Architecture1D import MyLinearLayerBCConfMore, BC_shima  # Custom architecture layers
from SRC.LSArchitecture import update_u  # Function to update model parameters
from SRC.data import u_exact  # Exact solution function
from SRC.Test_functions import test_functions  # Test functions for model

# Set the default data type to float64 for TensorFlow operations
dtype = "float64"
tf.keras.backend.set_floatx(dtype)

# Define custom colors for visualization
My_blue = (0, 0/255, 102/255)
My_Green = (0/255, 0/255, 102/255)
My_Orange = (255/255, 178/255, 102/255)
My_Green_Light = (146/255, 220/255, 181/255)

# Epsilon for small perturbations (unused)
eps = 0

# Load initial data from a saved .mat file
Data = scipy.io.loadmat('Saved_Models/initial_data.mat')
npts = Data['npts'].item()  # Number of integration points
eps_npts = Data['eps_npts'].item()  # Additional integration points for validation
max_p1 = Data['max_p1'].item() - eps  # Max value for parameter p1
min_p1 = Data['min_p1'].item() + eps  # Min value for parameter p1
max_p2 = Data['max_p2'].item() - eps  # Max value for parameter p2
min_p2 = Data['min_p2'].item() + eps  # Min value for parameter p2
max_p3 = Data['max_p3'].item() - eps  # Max value for parameter p3
min_p3 = Data['min_p3'].item() + eps  # Min value for parameter p3
dat_len = Data['dat_len'].item()  # Length of dataset
neurons_final = Data['neurons_final'].item()  # Number of neurons in the final layer
nmodes = Data['nmodes'].item()  # Number of modes

# Generate test functions for model training and validation
x, DST, DCT, DCT_a, DCT_b, norms = test_functions(nmodes, npts)
x_val, DST_val, DCT_val, DCT_a_val, DCT_b_val, norms_val = test_functions(nmodes, npts + eps_npts)

# Generate a test set for evaluation
xtest = tf.constant([0 + i*(1 - 0)/1000 for i in range(1000)], dtype=dtype)

################################################################
################################################################
################################################################
################################################################
# Register custom layers for loading pre-trained models
get_custom_objects().update({
    "MyLinearLayerBCConfMore": MyLinearLayerBCConfMore,
    "BC_shima": BC_shima,
})

# Load models before training
u_model = keras.models.load_model('Saved_Models/u_model_before_training.keras', compile=False)
u_bases = keras.models.load_model('Saved_Models/u_bases_before_training.keras', compile=False)

################################################################
################################################################
################################################################
################################################################
# Initializing lists for plotting results before training

xtest = tf.constant([((i+0.5)/1000) for i in range(1000)], dtype=dtype)  # Test points
l2_list_D_B = []  # List to store relative errors before training
h1_list_D_B = []  # List to store gradient errors (currently commented out)

# Perform predictions and compute errors before training
for i in range(1000):
    # Generate random test parameters
    Ptest_1 = (tf.random.uniform([1], dtype=dtype, minval=10**min_p1, maxval=10**max_p1))
    Ptest_2 = (tf.random.uniform([1], dtype=dtype, minval=10**min_p2, maxval=10**max_p2))
    Ptest_3 = (tf.random.uniform([1], dtype=dtype, minval=min_p3, maxval=max_p3))
    
    # Update model with current parameters
    cond = update_u(u_model, u_bases, tf.stack([Ptest_1, Ptest_2, Ptest_3], axis=-1), neurons_final, x, DST, DCT, DCT_a, DCT_b, norms)
    
    # Compute gradients of predicted and exact solutions
    with tf.GradientTape(persistent=True) as t1:
        t1.watch(xtest)
        u_r = u_model(xtest)[:, 0]
        u_i = u_model(xtest)[:, 1]
        ue_r = u_exact(xtest, Ptest_1, Ptest_2, Ptest_3)[:, 0]
        ue_i = u_exact(xtest, Ptest_1, Ptest_2, Ptest_3)[:, 1]
    
    du_r = t1.gradient(u_r, xtest)
    du_i = t1.gradient(u_i, xtest)
    due_r = t1.gradient(ue_r, xtest)
    due_i = t1.gradient(ue_i, xtest)
    
    # Compute L2 relative error and append to list
    l2_list_D_B += [float(tf.reduce_sum((ue_r-u_r)**2 + (ue_i-u_i)**2 + \
                                        (due_r-du_r)**2 + (due_i-du_i)**2)/\
                          tf.reduce_sum(ue_r**2 + ue_i**2 + due_r**2 + due_i**2))**0.5*100]

################################################################
################################################################
# Load models after training
################################################################
################################################################

# Register custom layers again for loading trained models
get_custom_objects().update({
    "MyLinearLayerBCConfMore": MyLinearLayerBCConfMore,
    "BC_shima": BC_shima,
})

# Load trained models
u_model = keras.models.load_model('Saved_Models/u_model_after_training.keras', compile=False)
u_bases = keras.models.load_model('Saved_Models/u_bases_after_training.keras', compile=False)

################################################################
################################################################
# Perform predictions and compute errors after training

l2_list_D_A = []  # List to store relative errors after training
h1_list_D_A = []  # List to store gradient errors (currently commented out)

for i in range(1000):
    # Generate random test parameters
    Ptest_1 = (tf.random.uniform([1], dtype=dtype, minval=10**min_p1, maxval=10**max_p1))
    Ptest_2 = (tf.random.uniform([1], dtype=dtype, minval=10**min_p2, maxval=10**max_p2))
    Ptest_3 = (tf.random.uniform([1], dtype=dtype, minval=min_p3, maxval=max_p3))
    
    # Update model with current parameters
    cond = update_u(u_model, u_bases, tf.stack([Ptest_1, Ptest_2, Ptest_3], axis=-1), neurons_final, x, DST, DCT, DCT_a, DCT_b, norms)
    
    # Compute gradients of predicted and exact solutions
    with tf.GradientTape(persistent=True) as t1:
        t1.watch(xtest)
        u_r = u_model(xtest)[:, 0]
        u_i = u_model(xtest)[:, 1]
        ue_r = u_exact(xtest, Ptest_1, Ptest_2, Ptest_3)[:, 0]
        ue_i = u_exact(xtest, Ptest_1, Ptest_2, Ptest_3)[:, 1]
    
    du_r = t1.gradient(u_r, xtest)
    du_i = t1.gradient(u_i, xtest)
    due_r = t1.gradient(ue_r, xtest)
    due_i = t1.gradient(ue_i, xtest)
    
    # Compute L2 relative error and append to list
    l2_list_D_A += [float(tf.reduce_sum((ue_r-u_r)**2 + (ue_i-u_i)**2 + \
                                        (due_r-du_r)**2 + (due_i-du_i)**2)/\
                          tf.reduce_sum(ue_r**2 + ue_i**2 + due_r**2 + due_i**2))**0.5*100]

################################################################
################################################################
# Plot histograms of the errors before and after training
################################################################

# Set font properties for the plot
font = {'family': 'Times',
        'weight': 'normal',
        'size': 15}
plt.rc('font', **font)

# Create a histogram plot for relative errors
plt.figure(figsize=(6.5, 3.2))
plt.rcParams['text.usetex'] = True
A1 = plt.hist(l2_list_D_A, bins=10**np.linspace(-3, np.log10(0.7) + 2, 60), color=My_Green, edgecolor='white', label='After training', alpha=0.8)
plt.hist(l2_list_D_B, bins=10**np.linspace(-3, np.log10(0.7) + 2, 60), color=My_Orange, edgecolor='white', label='Before training', alpha=0.7)

# Set logarithmic scales for x and y axes
plt.xscale('log')
plt.yscale('log')

# Set labels and legend
plt.xlabel(r"Relative $\textit{H}^1$ error of $u^{p, \alpha}$ (\%)")
plt.ylabel("Number of occurrences")
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False)

# Adjust plot spacing
plt.subplots_adjust(left=0.2,
                    bottom=0.19,
                    right=0.95,
                    top=0.85,
                    wspace=0.1,
                    hspace=0.2)

# Save the plot as .pdf
plt.savefig('Results/L2-PINN_D_B.pdf', dpi=300)

# Show the plot
plt.show()

################################################################
################################################################
# Plot figures for two specific sets of parameters
################################################################
################################################################

# Define test points between 0.01 and 0.99, spaced evenly
xtest = tf.constant([0.01 + i*(1 - 0.02)/999 for i in range(1000)], dtype=dtype)

# Set the first specific set of parameters
Ptest_1 = tf.constant([0.08], dtype=dtype)
Ptest_2 = tf.constant([0.1], dtype=dtype)
Ptest_3 = tf.constant([10], dtype=dtype)

# Update the model with the chosen parameters
cond = update_u(u_model, u_bases, tf.stack([Ptest_1, Ptest_2, Ptest_3], axis=-1), neurons_final, x, DST, DCT, DCT_a, DCT_b, norms)

# Compute the predicted solution (real and imaginary parts)
u_r = u_model(xtest)[:, 0]  # Real part of predicted solution
u_i = u_model(xtest)[:, 1]  # Imaginary part of predicted solution

# Compute the exact solution (real and imaginary parts)
ue_r = u_exact(xtest, Ptest_1, Ptest_2, Ptest_3)[:, 0]  # Real part of exact solution
ue_i = u_exact(xtest, Ptest_1, Ptest_2, Ptest_3)[:, 1]  # Imaginary part of exact solution

# Set font properties for the plot
font = {'family': 'Times', 'weight': 'normal', 'size': 15}
plt.rc('font', **font)
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{mathrsfs}'

# Create the subplots with the desired figure size
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

# Adjust spacing between subplots (optional)
plt.subplots_adjust(left=0.11, bottom=0.2, right=0.95, top=0.8, wspace=0.4, hspace=0.1)

# Plot for the real part
axes[0].plot(xtest, ue_r, label=r"Exact solution", linewidth=2, color=My_Green)  # Exact solution
axes[0].plot(xtest, u_r, label=r"$u^{p, \alpha}$", linewidth=1.5, linestyle='--', color=My_Orange)  # Predicted solution
axes[0].set_xlabel(r'$x$')
axes[0].set_ylabel('Real value')

# Plot for the imaginary part
axes[1].plot(xtest, ue_i, label=r"Exact solution", linewidth=2, color=My_Green)  # Exact solution
axes[1].plot(xtest, u_i, label=r"$u^{p, \alpha}$", linewidth=1.5, linestyle='--', color=My_Orange)  # Predicted solution
axes[1].set_xlabel(r'$x$')
axes[1].set_ylabel('Imaginary value')

# Extract handles and labels for the legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.8), fontsize=17, frameon=False)

# Save the plot as .pdf
plt.savefig('Results/complex_high_fre.pdf', dpi=300)

# Show the plot
plt.show()

################################################################
# Second specific set of parameters
################################################################

# Set the second specific set of parameters
Ptest_1 = tf.constant([50], dtype=dtype)
Ptest_2 = tf.constant([0.05], dtype=dtype)
Ptest_3 = tf.constant([5], dtype=dtype)

# Update the model with the new set of parameters
cond = update_u(u_model, u_bases, tf.stack([Ptest_1, Ptest_2, Ptest_3], axis=-1), neurons_final, x, DST, DCT, DCT_a, DCT_b, norms)

# Compute the predicted solution (real and imaginary parts)
u_r = u_model(xtest)[:, 0]  # Real part of predicted solution
u_i = u_model(xtest)[:, 1]  # Imaginary part of predicted solution

# Compute the exact solution (real and imaginary parts)
ue_r = u_exact(xtest, Ptest_1, Ptest_2, Ptest_3)[:, 0]  # Real part of exact solution
ue_i = u_exact(xtest, Ptest_1, Ptest_2, Ptest_3)[:, 1]  # Imaginary part of exact solution

# Set font properties for the plot
font = {'family': 'Times', 'weight': 'normal', 'size': 15}
plt.rc('font', **font)
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{mathrsfs}'

# Create the subplots with the desired figure size
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

# Adjust spacing between subplots (optional)
plt.subplots_adjust(left=0.11, bottom=0.2, right=0.95, top=0.8, wspace=0.4, hspace=0.1)

# Plot for the real part
axes[0].plot(xtest, ue_r, label=r"Exact solution", linewidth=2, color=My_Green)  # Exact solution
axes[0].plot(xtest, u_r, label=r"$u^{p, \alpha}$", linewidth=1.5, linestyle='--', color=My_Orange)  # Predicted solution
axes[0].set_xlabel(r'$x$')
axes[0].set_ylabel('Real value')

# Plot for the imaginary part
axes[1].plot(xtest, ue_i, label=r"Exact solution", linewidth=2, color=My_Green)  # Exact solution
axes[1].plot(xtest, u_i, label=r"$u^{p, \alpha}$", linewidth=1.5, linestyle='--', color=My_Orange)  # Predicted solution
axes[1].set_xlabel(r'$x$')
axes[1].set_ylabel('Imaginary value')

# Extract handles and labels for the legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.8), fontsize=17, frameon=False)

# Save the plot as .pdf
plt.savefig('Results/complex_low_fre.pdf', dpi=300)

# Show the plot
plt.show()

################################################################
################################################################
# Plotting the loss function during training and validation
################################################################
################################################################

# Load training data history from a .mat file
data = scipy.io.loadmat('Saved_Models/Data_history_1.mat')
iterations = int(data['iterations'])  # Number of iterations
history = np.concatenate(data['history'])  # Training loss
history_val = np.concatenate(data['history_val'])  # Validation loss

# Set font properties for the plot
font = {'family': 'Times', 'weight': 'normal', 'size': 15}
plt.rc('font', **font)

# Create figure for the loss plot
plt.figure(figsize=(6.5, 3.2))
plt.rcParams['text.usetex'] = True

# Plot training loss
plt.plot([(i) for i in range(1, iterations+1)], history, '-', color=My_Orange, linewidth=1.5, label='Training')

# Plot validation loss
plt.plot([(i) for i in range(1, iterations+1)], history_val, '-.', color=My_Green, linewidth=1.5, label='Validation')

# Set labels and legend
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=2, frameon=False)
plt.xlabel('Iteration')
plt.ylabel(r"Loss")
plt.xscale("log")
plt.yscale("log")

# Set x-ticks with logarithmic labels
plt.xticks([1, 10, 100, 1000, 10000], ['$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'])

# Adjust plot layout
plt.subplots_adjust(left=0.2, bottom=0.19, right=0.95, top=0.85, wspace=0.1, hspace=0.2)

# Save the loss plot as .pdf
plt.savefig('Results/Loss_0.pdf', dpi=300)

# Show the plot
plt.show()
