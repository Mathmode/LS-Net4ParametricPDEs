#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 21:47:19 2024

@author: sbahalouei
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.io
import keras
###############################################
from SRC.Collocation_points import collocation_points
from SRC.LSArchitecture import update_u
from SRC.ProblemDamped import u_exact

dtype="float64"
tf.keras.backend.set_floatx(
    dtype
)


My_blue=(0, 0/255, 102/255)
My_Green = (0/255,0/255,102/255)
My_Orange = (255/255, 178/255, 102/255)
My_Green_Light = (146/255, 220/255, 181/255)

eps = 0

Data = scipy.io.loadmat('Saved_Models/initial_data.mat')
a = Data['a'].item()
b = Data['b'].item()
x0 = Data['x0'].item()
v0 = Data['v0'].item()
npts = Data['npts'].item()
eps_npts = Data['eps_npts'].item()
max_p1 = Data['max_p1'].item() - eps
min_p1 = Data['min_p1'].item() + eps
max_p2 = Data['max_p2'].item() - eps
min_p2 = Data['min_p2'].item() + eps
dat_len = Data['dat_len'].item()
neurons_final = Data['neurons_final'].item()


x = collocation_points(a, b, npts)
n = 80
xtest = tf.constant([a + i*(b - a)/1000 for i in range(1000)],dtype=dtype)

################################################################
################################################################
################################################################
################################################################

u_model = keras.models.load_model('Saved_Models/u_model_before_training.keras', compile=False)
u_bases = keras.models.load_model('Saved_Models/u_bases_before_training.keras', compile=False)

################################################################
################################################################
################################################################
################################################################


################################################################
################################################################
# initializing for plotting before training
################################################################
################################################################
h1_list_B = np.zeros((n, n)) 
h2_list_B = np.zeros((n, n)) 
h3_list_B = np.zeros((n, n)) 
l2_list_B = np.zeros((n, n)) 
#pw_list_PINN = np.zeros((n, n))
Ptest_1 = tf.linspace(10**min_p1, 10**max_p1, n)[:, tf.newaxis]
Ptest_2 = tf.linspace(10**min_p2, 10**max_p2, n)[:, tf.newaxis]
Ptest_1 = tf.cast(Ptest_1, tf.float64)
Ptest_2 = tf.cast(Ptest_2, tf.float64)
X, Y = np.meshgrid(Ptest_1, Ptest_2)



k1 = n - 1

for i in range(n):
    for j in range(n):
        cond = update_u(u_model, u_bases, tf.stack([Ptest_1[j], Ptest_2[i]], axis=-1), neurons_final, x0, v0, x, npts)
        with tf.GradientTape(persistent=True) as t1:
            with tf.GradientTape(persistent=True) as t2:
                with tf.GradientTape(persistent=True) as t3:
                    t3.watch(xtest)
                    t1.watch(xtest)
                    t2.watch(xtest)
                    u = u_model(xtest)
                    ue = u_exact(xtest, Ptest_1[j], Ptest_2[i], x0, v0)

                due = t3.gradient(ue,xtest)
                du = t3.gradient(u,xtest)
            ddue = t2.gradient(due,xtest)
            ddu = t2.gradient(du,xtest)
        dddue = t1.gradient(ddue,xtest)
        dddu = t1.gradient(ddu,xtest)
        l2_list_B[k1, j] = float(tf.reduce_sum((ue-u)**2)/tf.reduce_sum(ue**2))**0.5*100
        h1_list_B[k1, j] = float(tf.reduce_sum((due-du)**2)/tf.reduce_sum(due**2))**0.5*100
        h2_list_B[k1, j] = float(tf.reduce_sum((ddue-ddu)**2)/tf.reduce_sum(ddue**2))**0.5*100
        h3_list_B[k1, j] = float(tf.reduce_sum((dddue-dddu)**2)/tf.reduce_sum(dddue**2))**0.5*100

#        pw_list_PINN[k1, j] = float(tf.reduce_sum(tf.abs(ue-u_PINN)))*100
    k1 = k1 - 1       
#####################################################
# plot distributions before training
#pw_list_PINN_D = []
h1_list_D_B = []
h2_list_D_B = []
h3_list_D_B = []
l2_list_D_B = []

for i in range(200):
    Ptest_1 = (tf.random.uniform([1],dtype=dtype,minval=10**min_p1,maxval=10**max_p1))
    Ptest_2 = (tf.random.uniform([1],dtype=dtype,minval=10**min_p2,maxval=10**max_p2))
    cond = update_u(u_model, u_bases, tf.stack([Ptest_1, Ptest_2], axis=-1), neurons_final, x0, v0, x, npts)
    with tf.GradientTape(persistent=True) as t1:
        with tf.GradientTape(persistent=True) as t2:
            with tf.GradientTape(persistent=True) as t3:
                t3.watch(xtest)
                t2.watch(xtest)
                t1.watch(xtest)
                u = u_model(xtest)
                ue = u_exact(xtest, Ptest_1, Ptest_2, x0, v0)
            due = t3.gradient(ue,xtest)
            du = t3.gradient(u,xtest)
        ddue = t2.gradient(due,xtest)
        ddu = t2.gradient(du,xtest)
    dddue = t1.gradient(ddue,xtest)
    dddu = t1.gradient(ddu,xtest)
    
#    pw_list_PINN_D +=[float(tf.reduce_sum(tf.abs(ue-u_PINN)))]
    l2_list_D_B += [float(tf.reduce_sum((ue-u)**2)/tf.reduce_sum(ue**2))**0.5*100]
    h1_list_D_B += [float(tf.reduce_sum((due-du)**2)/tf.reduce_sum(due**2))**0.5*100]
    h2_list_D_B += [float(tf.reduce_sum((ddue-ddu)**2)/tf.reduce_sum(ddue**2))**0.5*100]
    h3_list_D_B += [float(tf.reduce_sum((dddue-dddu)**2)/tf.reduce_sum(dddue**2))**0.5*100]
################################################################
################################################################
# plot Figures before training
################################################################
################################################################



import matplotlib.colors as mcolors
font = {'family' : 'Times',
        'weight' : 'normal',
        'size'   : 15}
plt.rc('font', **font)
plt.rcParams['text.usetex'] = True
# Define the range for x and y
x_range = [min_p1, max_p1]
y_range = [min_p2 ,max_p2]

vmin = min(l2_list_B.min(), h1_list_B.min(), h2_list_B.min())
vmax = max(l2_list_B.max(), h1_list_B.max(), h2_list_B.max())
log_vmin, log_vmax = ([vmin, vmax])
# Define the colormap and normalization
#cmap = 'coolwarm'

# Define a custom colormap
colors = [My_Orange, 'white', My_Green]  # Blue -> Cyan -> Green -> Yellow -> Red
cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors)


norm = LogNorm(vmin=log_vmin, vmax=log_vmax)
# Create the figure and axes
fig, axes = plt.subplots(1, 3, figsize=(10, 4))

im1 = axes[0].imshow(l2_list_B,interpolation='bilinear', extent=[x_range[0], x_range[1], y_range[0], y_range[1]], aspect='auto', cmap=cmap, norm=norm)
axes[0].set_title(r"Relative $\textit{L}^2$ error of $u^{p, \alpha}$ ( \%)", fontsize=15)
im2 = axes[1].imshow(h1_list_B,interpolation='bilinear', extent=[x_range[0], x_range[1], y_range[0], y_range[1]], aspect='auto', cmap=cmap, norm=norm)
axes[1].set_title(r"Relative $\textit{L}^2$ error of $\displaystyle u^{p, \alpha}_{t}$  (\%)", fontsize=15)
im2 = axes[2].imshow(h2_list_B,interpolation='bilinear', extent=[x_range[0], x_range[1], y_range[0], y_range[1]], aspect='auto', cmap=cmap, norm=norm)
axes[2].set_title(r"Relative $\textit{L}^2$ error of $\displaystyle u^{p, \alpha}_{tt}$  (\%)", fontsize=15)

# Add a dedicated Axes for the colorbar
cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.03])  # [left, bottom, width, height]

# Add a single colorbar
cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal', fraction=0.03, pad=0.1)

p_1 = np.linspace(min_p1, max_p1, 400)
p_2 = -p_1 -np.log10(4)


for ax in axes:
    # Adding text annotation
    ax.text(-0.6, -1, 'Underdamped', fontsize=15, color='black', ha='center', va='center')
    ax.text(0.6, -0.2, 'Overdamped', fontsize=15, color='black', ha='center', va='center')
    ax.arrow(-.7, 0.5, -0.1, -0.1, head_width=0.07, head_length=.05, fc='black', ec='black')
    ax.text(0.2, 0.6, 'Critically Damped', fontsize=15, color='black', ha='center', va='center')

    ax.plot(p_1, p_2, color='red')
    ax.set_xlim(min_p1, max_p1)
    ax.set_ylim(min_p2, max_p2)
    
    ax.set_xlabel(r'$\log _{10}(p_1)$', labelpad=.2)  # Label for x-axis
    ax.set_ylabel(r'$\log _{10}(p_2)$', labelpad=.2)  # Label for y-axis
    ax.xaxis.set_label_coords(0.5, -0.12)  # Adjust the y-value for fine-tuning x-label position
    ax.yaxis.set_label_coords(-0.1, 0.5)  # Adjust the x-value for fine-tuning y-label position


plt.subplots_adjust(left=0.05,
                    bottom=0.27,
                    right=0.95,
                    top=0.85,
                    wspace=0.2,
                    hspace=0.2)

plt.savefig('Results/L2_error_PINN_BT.pdf',dpi=300)#for graphs
plt.show()






################################################################
################################################################
# Load the model after training
################################################################
################################################################
u_model = keras.models.load_model('Saved_Models/u_model_after_training.keras', compile=False)
u_bases = keras.models.load_model('Saved_Models/u_bases_after_training.keras', compile=False)
################################################################
################################################################

################################################################
################################################################
# initializing for plotting before training
################################################################
################################################################

h1_list_A = np.zeros((n, n)) 
h2_list_A = np.zeros((n, n)) 
h3_list_A = np.zeros((n, n)) 
l2_list_A = np.zeros((n, n)) 
#pw_list_PINN = np.zeros((n, n))
Ptest_1 = tf.linspace(10**min_p1, 10**max_p1, n)[:, tf.newaxis]
Ptest_2 = tf.linspace(10**min_p2, 10**max_p2, n)[:, tf.newaxis]
Ptest_1 = tf.cast(Ptest_1, tf.float64)
Ptest_2 = tf.cast(Ptest_2, tf.float64)
X, Y = np.meshgrid(Ptest_1, Ptest_2)

k1 = n - 1

for i in range(n):
    for j in range(n):
        cond= update_u(u_model, u_bases, tf.stack([Ptest_1[j], Ptest_2[i]], axis=-1), neurons_final, x0, v0, x, npts)
        with tf.GradientTape(persistent=True) as t1:
            with tf.GradientTape(persistent=True) as t2:
                with tf.GradientTape(persistent=True) as t3:
                    t3.watch(xtest)
                    t1.watch(xtest)
                    t2.watch(xtest)
                    u = u_model(xtest)
                    ue = u_exact(xtest, Ptest_1[j], Ptest_2[i], x0, v0)

                due = t3.gradient(ue,xtest)
                du = t3.gradient(u,xtest)
            ddue = t2.gradient(due,xtest)
            ddu = t2.gradient(du,xtest)
        dddue = t1.gradient(ddue,xtest)
        dddu = t1.gradient(ddu,xtest)
        l2_list_A[k1, j] = float(tf.reduce_sum((ue-u)**2)/tf.reduce_sum(ue**2))**0.5*100
        h1_list_A[k1, j] = float(tf.reduce_sum((due-du)**2)/tf.reduce_sum(due**2))**0.5*100
        h2_list_A[k1, j] = float(tf.reduce_sum((ddue-ddu)**2)/tf.reduce_sum(ddue**2))**0.5*100
        h3_list_A[k1, j] = float(tf.reduce_sum((dddue-dddu)**2)/tf.reduce_sum(dddue**2))**0.5*100

#        pw_list_PINN[k1, j] = float(tf.reduce_sum(tf.abs(ue-u_PINN)))*100
    k1 = k1 - 1       
#####################################################

# plot distributions before training
#pw_list_PINN_D = []
h1_list_D_A = []
h2_list_D_A = []
h3_list_D_A = []
l2_list_D_A = []

for i in range(200):
    Ptest_1 = (tf.random.uniform([1],dtype=dtype,minval=10**min_p1,maxval=10**max_p1))
    Ptest_2 = (tf.random.uniform([1],dtype=dtype,minval=10**min_p2,maxval=10**max_p2))
    
    cond = update_u(u_model, u_bases, tf.stack([Ptest_1, Ptest_2], axis=-1), neurons_final, x0, v0, x, npts)
    with tf.GradientTape(persistent=True) as t1:
        with tf.GradientTape(persistent=True) as t2:
            with tf.GradientTape(persistent=True) as t3:
                t3.watch(xtest)
                t2.watch(xtest)
                t1.watch(xtest)
                u = u_model(xtest)
                ue = u_exact(xtest, Ptest_1, Ptest_2, x0, v0)
            due = t3.gradient(ue,xtest)
            du = t3.gradient(u,xtest)
        ddue = t2.gradient(due,xtest)
        ddu = t2.gradient(du,xtest)
    dddue = t1.gradient(ddue,xtest)
    dddu = t1.gradient(ddu,xtest)
    
#    pw_list_PINN_D +=[float(tf.reduce_sum(tf.abs(ue-u_PINN)))]
    l2_list_D_A += [float(tf.reduce_sum((ue-u)**2)/tf.reduce_sum(ue**2))**0.5*100]
    h1_list_D_A += [float(tf.reduce_sum((due-du)**2)/tf.reduce_sum(due**2))**0.5*100]
    h2_list_D_A += [float(tf.reduce_sum((ddue-ddu)**2)/tf.reduce_sum(ddue**2))**0.5*100]
    h3_list_D_A += [float(tf.reduce_sum((dddue-dddu)**2)/tf.reduce_sum(dddue**2))**0.5*100]

################################################################
################################################################
# plot Figures after training
################################################################
################################################################

import matplotlib.colors as mcolors
font = {'family' : 'Times',
        'weight' : 'normal',
        'size'   : 15}
plt.rc('font', **font)
plt.rcParams['text.usetex'] = True
# Define the range for x and y
x_range = [min_p1, max_p1]
y_range = [min_p2 ,max_p2]

vmin = min(l2_list_A.min(), h1_list_A.min(), h2_list_A.min())
vmax = max(l2_list_A.max(), h1_list_A.max(), h2_list_A.max())
log_vmin, log_vmax = ([vmin, vmax])

colors = [My_Orange, 'white', My_Green]  # Blue -> Cyan -> Green -> Yellow -> Red
cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors)


norm = LogNorm(vmin=log_vmin, vmax=log_vmax)
# Create the figure and axes
fig, axes = plt.subplots(1, 3, figsize=(10, 4))

im1 = axes[0].imshow(l2_list_A,interpolation='bilinear', extent=[x_range[0], x_range[1], y_range[0], y_range[1]], aspect='auto', cmap=cmap, norm=norm)
axes[0].set_title(r"Relative $\textit{L}^2$ error of $u^{p, \alpha}$ (\%)\vspace{0.5cm}", fontsize=15)
im2 = axes[1].imshow(h1_list_A,interpolation='bilinear', extent=[x_range[0], x_range[1], y_range[0], y_range[1]], aspect='auto', cmap=cmap, norm=norm)
axes[1].set_title(r"Relative $\textit{L}^2$ error of $ \displaystyle u^{p, \alpha}_{t}$ (\%)", fontsize=15)
im2 = axes[2].imshow(h2_list_A,interpolation='bilinear', extent=[x_range[0], x_range[1], y_range[0], y_range[1]], aspect='auto', cmap=cmap, norm=norm)
axes[2].set_title(r"Relative $\textit{L}^2$ error of $\displaystyle u^{p, \alpha}_{tt}$ (\%)", fontsize=15)

# Add a dedicated Axes for the colorbar
cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.03])  # [left, bottom, width, height]

# Add a single colorbar
cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal', fraction=0.03, pad=0.1)


p_1 = np.linspace(min_p1, max_p1, 400)
p_2 = -p_1 -np.log10(4)


for ax in axes:
    # Adding text annotation
    ax.text(-0.6, -1, 'Underdamped', fontsize=15, color='black', ha='center', va='center')
    ax.text(0.6, -0.2, 'Overdamped', fontsize=15, color='black', ha='center', va='center')
    ax.arrow(-.7, 0.5, -0.1, -0.1, head_width=0.07, head_length=.05, fc='black', ec='black')
    ax.text(0.2, 0.6, 'Critically Damped', fontsize=15, color='black', ha='center', va='center')

    ax.plot(p_1, p_2, color='red')
    ax.set_xlim(min_p1, max_p1)
    ax.set_ylim(min_p2, max_p2)
    
    ax.set_xlabel(r'$\log_{10}(p_1)$', labelpad=.2)
    ax.set_ylabel(r'$\log_{10}(p_2)$', labelpad=.2)
    ax.xaxis.set_label_coords(0.5, -0.12)
    ax.yaxis.set_label_coords(-0.1, 0.5)  

plt.subplots_adjust(left=0.05,
                    bottom=0.27,
                    right=0.95,
                    top=0.85,
                    wspace=0.2,
                    hspace=0.2)

plt.savefig('Results/L2_error_PINN_AT.pdf',dpi=300)
plt.show()



################################################################
################################################################
# Distribution plots
################################################################
################################################################


# Font settings
font = {'family': 'Times',
        'weight': 'normal',
        'size': 17}
plt.rc('font', **font)
plt.rcParams['text.usetex'] = True

# Create the subplots
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

# Adjust spacing between subplots
fig.subplots_adjust(left=0.07, bottom=0.27, right=0.98, top=0.85, wspace=0.3, hspace=0.1)

# Plot the histograms
axes[0].hist(l2_list_D_A, bins=10**np.linspace(-6, 3, 50), color=My_Green, edgecolor='white', label='After training', alpha = 0.8)
axes[0].hist(l2_list_D_B, bins=10**np.linspace(-6, 3, 50), color=My_Orange, edgecolor='white', label='Before training', alpha = 0.8)
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_xlabel(r"Relative $\textit{L}^2$ error of $\displaystyle u^{p, \alpha}$  (\%)")
axes[0].set_ylabel("Number of occurrences")

axes[1].hist(h1_list_D_A, bins=10**np.linspace(-5, 3, 50), color=My_Green, edgecolor='white', label='After training', alpha = 0.8)
axes[1].hist(h1_list_D_B, bins=10**np.linspace(-5, 3, 50), color=My_Orange, edgecolor='white', label='Before training', alpha = 0.8)
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].set_xlabel(r"Relative $\textit{L}^2$ error of $\displaystyle u^{p, \alpha}_{t}$   (\%)")
axes[1].set_ylabel("Number of occurrences")

axes[2].hist(h2_list_D_A, bins=10**np.linspace(-4, 2, 50), color=My_Green, edgecolor='white', label='After training', alpha = 0.8)
axes[2].hist(h2_list_D_B, bins=10**np.linspace(-4, 2, 50), color=My_Orange, edgecolor='white', label='Before training', alpha = 0.8)
axes[2].set_xscale('log')
axes[2].set_yscale('log')
axes[2].set_xlabel(r"Relative $\textit{L}^2$ error of $\displaystyle u^{p, \alpha}_{tt}$   (\%)")
axes[2].set_ylabel("Number of occurrences")

handles, labels = axes[0].get_legend_handles_labels()

# Create a common legend below all subplots
fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.84), fontsize=17, frameon=False)

# Save and show the plot
plt.savefig('Results/L2_PINN_D.pdf', dpi=300)
plt.show()




################################################################
################################################################
# plotting three cases
################################################################
################################################################


xtest = tf.constant([a + i*(b - a)/1000 for i in range(1000)],dtype=dtype)



Ptest_1 = tf.constant([0.03], dtype=dtype)
Ptest_2 = tf.constant([1], dtype=dtype)

Ptest_3 = tf.constant([1], dtype=dtype)
Ptest_4 = tf.constant([1/4], dtype=dtype)

Ptest_5 = tf.constant([2], dtype=dtype)
Ptest_6 = tf.constant([5], dtype=dtype)

Ptest_7 = tf.constant([10**(-1.5)], dtype=dtype)
Ptest_8 = tf.constant([10**(-1.5)], dtype=dtype)

cond = update_u(u_model, u_bases, tf.stack([Ptest_1, Ptest_2], axis=-1), neurons_final, x0, v0, x, npts)
u_1 = u_model(xtest)
ue_1 = u_exact(xtest, Ptest_1, Ptest_2, x0, v0)

cond = update_u(u_model, u_bases, tf.stack([Ptest_3, Ptest_4], axis=-1), neurons_final, x0, v0, x, npts)
u_2 = u_model(xtest)
ue_2 = u_exact(xtest, Ptest_3, Ptest_4, x0, v0)

cond = update_u(u_model, u_bases, tf.stack([Ptest_5, Ptest_6], axis=-1), neurons_final, x0, v0, x, npts)
u_3 = u_model(xtest)
ue_3 = u_exact(xtest, Ptest_5, Ptest_6, x0, v0)

cond = update_u(u_model, u_bases, tf.stack([Ptest_7, Ptest_8], axis=-1), neurons_final, x0, v0, x, npts)
u_4 = u_model(xtest)
ue_4 = u_exact(xtest, Ptest_7, Ptest_8, x0, v0)




font = {'family': 'Times',
        'weight': 'normal',
        'size': 15}
plt.rc('font', **font)
plt.rc('text', usetex=True)
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath} \usepackage{mathrsfs}'


# Create the subplots with the desired figure size
fig, axes = plt.subplots(3, 1, figsize=(9, 7))

# Adjust spacing between subplots (optional)
plt.subplots_adjust(left=.11, bottom=0.1, right=0.95, top=0.88, wspace=0.4, hspace=0.75)

axes[0].plot(xtest, ue_1, label=r"Exact solution", linewidth=2, color=My_Green)
axes[0].plot(xtest, u_1, label=r"$u^{p, \alpha}$", linewidth=1.5, linestyle='--', color=My_Orange)
axes[0].set_xlabel(r'$x$')
axes[0].set_ylabel('value')
axes[0].set_title(rf"Overdamped with $p=({Ptest_1[0]}, {int(Ptest_2[0])})$", fontsize=15)

axes[1].plot(xtest, ue_2, label=r"Exact solution", linewidth=2, color=My_Green)
axes[1].plot(xtest, u_2, label=r"$u^{p, \alpha}$", linewidth=1.5, linestyle='--', color=My_Orange)
axes[1].set_xlabel(r'$x$')
axes[1].set_ylabel('value')
axes[1].set_title(rf"Critically damped with $p=({int(Ptest_3[0])}, {Ptest_4[0]})$", fontsize=15)

axes[2].plot(xtest, ue_3, label=r"Exact solution", linewidth=2, color=My_Green)
axes[2].plot(xtest, u_3, label=r"$u^{p, \alpha}$", linewidth=1.5, linestyle='--', color=My_Orange)
axes[2].set_xlabel(r'$x$')
axes[2].set_ylabel('value')
axes[2].set_title(rf"Underdamped with $p=({int(Ptest_5[0])}, {int(Ptest_6[0])})$", fontsize=15)


# Extract handles and labels from the first axis (axes[0])
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1), fontsize=15, frameon=False)

# Save the figure
plt.savefig('Results/damped-different-parameters.pdf', dpi=300)

# Show the plots
plt.show()
    
################################################################
################################################################
# Plotting loss function and validation
################################################################
################################################################

data = scipy.io.loadmat('Saved_Models/Data_history_1.mat')
iterations = int(data['iterations'])
history = np.concatenate(data['history'])
history_val = np.concatenate(data['history_val'])

        
        
font = {'family' : 'Times',
        'weight' : 'normal',
        'size'   : 15}
plt.rc('font', **font)
plt.figure(figsize=(6.5, 3.2))
plt.rcParams['text.usetex'] = True

plt.plot([(i) for i in range(1,iterations+1)],history,'-', color=My_Orange, linewidth=1.5, label = 'Training')
plt.plot([(i) for i in range(1,iterations+1)],history_val, '-.',color=My_Green, linewidth=1.5, label = 'Validation')

#plt.title('PINN')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=2, frameon=False)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.xscale("log")
plt.yscale("log")
plt.xticks([1, 10, 100, 1000, 10000], ['$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'])#, r"$\displaystyle 5 \cdot 10^3$"])
plt.subplots_adjust(left  =0.2,
                    bottom=0.19,
                    right =0.95,
                    top   =0.85,
                    wspace=0.1,
                    hspace=0.2)
plt.savefig('Results/Loss_0.pdf',dpi=300)
plt.show()