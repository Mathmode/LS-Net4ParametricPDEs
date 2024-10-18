#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited on Sept, 2024

@author: curiarteb
"""

# Import necessary libraries for data handling and visualization
import pandas as pd, os, numpy as np
from SRC.config import Aparam, Bparam, MODELS_FOLDER, RESULTS_FOLDER

# Ensure Aparam and Bparam are integers
Aparam, Bparam = int(Aparam), int(Bparam)

# Set up for plotting and define custom colors
import matplotlib.pyplot as plt
font = {'family'   : 'Times',
        'weight' : 'normal',
        'size'   : 15}
plt.rc('font', **font)
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# Define custom colors for the plots
My_Blue = (0/255,0/255,102/255)
My_Orange = (255/255, 178/255, 102/255)
My_Red = (139/255, 0/255, 0/255)
My_Blue_1 = (40/255,40/255,142/255)
My_Orange_1 = (255/255, 178/255, 102/255)
My_Red_1 = (179/255, 32/255, 32/255)

# Create the results folder if it does not exist
if not os.path.exists(MODELS_FOLDER):
    os.makedirs(MODELS_FOLDER)

print()
print("#######################")
print(" POST-PROCESSED PLOTS  ")
print("#######################")
print()

# Read prediction data from CSV files
pred_constant = pd.read_csv(f"{MODELS_FOLDER}/pred_constant.csv", sep=";")
pred_materials = pd.read_csv(f"{MODELS_FOLDER}/pred_materials.csv", sep=";")

# Define custom colormap for the plots
import matplotlib.colors as mcolors
colors = [My_Orange,'white', My_Blue_1, My_Blue ,'black']  
cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors)

# Create the figure and axes for subplots
fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2))  

# Plot predictions with constant p (first subplot)
p1 = axes[0].scatter(pred_constant["x"], pred_constant["y"], c=pred_constant["pred"], cmap=cmap)
axes[0].set_xlabel(r"$x$", labelpad=5)
axes[0].set_ylabel(r"$y$", labelpad=5)
axes[0].set_title(rf"$u^{{p, \alpha}}$ with $p=({Aparam},{Aparam},{Aparam},{Aparam})$", fontsize=16)

# Plot predictions with non-constant p (second subplot)
p2 = axes[1].scatter(pred_materials["x"], pred_materials["y"], c=pred_materials["pred"], cmap=cmap)
axes[1].set_xlabel(r"$x$", labelpad=5)
axes[1].set_ylabel(r"$y$", labelpad=5)
axes[1].set_title(rf"$u^{{p, \alpha}}$ with $p=({Bparam},{Aparam},{Aparam},{Bparam})$", fontsize=16)

# Add a common colorbar below the subplots
cbar_ax = fig.add_axes([0.15, 0.12, 0.7, 0.03])  # Define position of the colorbar
cbar = fig.colorbar(p1, cax=cbar_ax, orientation='horizontal', fraction=0.03, pad=0.1)

# Adjust layout and save the figure
plt.subplots_adjust(left=0.11, bottom=0.3, right=0.94, top=0.85, wspace=0.3, hspace=0.2)
plt.savefig(f"{RESULTS_FOLDER}/predictions.pdf", dpi=300)
plt.show()

# Read histogram data for errors before and after training
hist_data = pd.read_csv(f"{MODELS_FOLDER}/histogram.csv", sep=";")

# Define font settings for the histograms
font = {'family': 'Times', 'weight': 'normal', 'size': 17}
plt.rc('font', **font)
plt.rcParams['text.usetex'] = True

# Create subplots for the histograms (before and after training)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
fig.subplots_adjust(left=0.07, bottom=0.27, right=0.98, top=0.85, wspace=0.3, hspace=0.1)

# Plot histogram for lower bounds of errors (first subplot)
axes[0].hist(hist_data["after_low"], bins=10**np.linspace(-0.7, 1, 50), color=My_Blue, edgecolor='white', label='After training', alpha=0.8)
axes[0].hist(hist_data["before_low"], bins=10**np.linspace(-0.7, 1, 50), color=My_Orange, edgecolor='white', label='Before training', alpha=0.8)
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_xlabel(r"Lower bounds for relative errors of $u^{p, \alpha}$ (\%)")
axes[0].set_ylabel("Number of occurrences")

# Plot histogram for upper bounds of errors (second subplot)
axes[1].hist(hist_data["after_up"], bins=10**np.linspace(0, 2, 50), color=My_Blue, edgecolor='white', label='After training', alpha=1)
axes[1].hist(hist_data["before_up"], bins=10**np.linspace(0, 2, 50), color=My_Orange, edgecolor='white', label='Before training', alpha=0.7)
axes[1].set_xscale('log')
axes[1].set_yscale('log')
axes[1].set_xlabel(r"Upper bounds for relative errors of $u^{p, \alpha}$ (\%)")
axes[1].set_ylabel("Number of occurrences")

# Extract handles and labels for the legend
handles, labels = axes[0].get_legend_handles_labels()

# Create a common legend above the subplots
fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.84), fontsize=17, frameon=False)

# Save and display the histograms
plt.savefig(f"{RESULTS_FOLDER}/histogram_transmition.pdf", bbox_inches='tight', dpi=300)
plt.show()






# Read loss data for training and validation
loss_data = pd.read_csv(f"{MODELS_FOLDER}/loss.csv", sep=";")
val_data = pd.read_csv(f"{MODELS_FOLDER}/validation.csv", sep=";")

# Plot the loss history
font = {'family': 'Times', 'weight': 'normal', 'size': 15}
plt.rc('font', **font)
plt.figure(figsize=(6.5, 3.2))
plt.rcParams['text.usetex'] = True

# Plot training loss
plt.plot(loss_data["iteration"], loss_data["loss"], '-', color=My_Blue, linewidth=1.5, label='Training')

# Plot validation losses (interior and truncated)
plt.plot(val_data["iteration"], val_data["loss_int"], 'o', color=My_Orange, linewidth=4, label='Int. Valid.', alpha=0.8)
plt.plot(val_data["iteration"], val_data["loss_test"], 'x', color=My_Red, linewidth=4, label='Trunc. Valid.', alpha=0.8)

# Set logarithmic scale for both axes
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')

# Set labels and title
plt.xlabel("Iterations")
plt.ylabel(r"Loss")

# Adjust layout for better appearance and save the loss plot
plt.subplots_adjust(left=0.15, bottom=0.19, right=0.95, top=0.85, wspace=0.1, hspace=0.2)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=3, frameon=False)
plt.savefig(f"{RESULTS_FOLDER}/loss.pdf", bbox_inches='tight', dpi=300)
plt.show()
