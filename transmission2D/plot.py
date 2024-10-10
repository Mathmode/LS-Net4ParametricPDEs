#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited on Sept, 2024

@author: curiarteb
"""

import pandas as pd, os, numpy as np

from config import Aparam, Bparam, RESULTS_FOLDER
Aparam, Bparam = int(Aparam),int(Bparam)

import matplotlib.pyplot as plt
font = {'family'   : 'Times',
        'weight' : 'normal',
        'size'   : 15}
plt.rc('font', **font)
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
My_Blue = (0/255,0/255,102/255)
My_Orange = (255/255, 178/255, 102/255)
My_Red = (139/255, 0/255, 0/255)

# Create the results folder if it does not exist
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

print()
print("#######################")
print(" POST-PROCESSED PLOTS  ")
print("#######################")
print()

# Define necessary variables and settings (e.g., RESULTS_FOLDER, My_Green, My_Orange, EXPERIMENT_REFERENCE, etc.)

# Read predictions
pred_constant = pd.read_csv(f"{RESULTS_FOLDER}/pred_constant.csv", sep=";")
pred_materials = pd.read_csv(f"{RESULTS_FOLDER}/pred_materials.csv", sep=";")

# Plot predictions with constant p
p = plt.scatter(pred_constant["x"], pred_constant["y"], c=pred_constant["pred"], cmap="plasma", alpha=0.7)
plt.colorbar(p)
plt.xlabel(r"$x$", labelpad=5)
plt.ylabel(r"$y$", labelpad=15)
plt.title(rf"Network prediction for $p=({{{Aparam}}},{{{Aparam}}},{{{Aparam}}},{{{Aparam}}})$")
plt.savefig(f"{RESULTS_FOLDER}/pred_constant.png", bbox_inches='tight', dpi=300)
plt.show()

# Plot predictions with non-constant p
p = plt.scatter(pred_materials["x"], pred_materials["y"], c=pred_materials["pred"], cmap="plasma", alpha=0.7)
plt.colorbar(p)
plt.xlabel(r"$x$", labelpad=5)
plt.ylabel(r"$y$", labelpad=15)
plt.title(rf"Network prediction for $p=({{{Bparam}}},{{{Aparam}}},{{{Aparam}}},{{{Bparam}}})$")
plt.savefig(f"{RESULTS_FOLDER}/pred_materials.png", bbox_inches='tight', dpi=300)
plt.show()

# Read histogram data
hist_data = pd.read_csv(f"{RESULTS_FOLDER}/histogram.csv", sep=";")

# Plot histograms
plt.hist(hist_data["after_up"], bins=10**np.linspace(-1, 2, 50), color=My_Blue, edgecolor='white', label='After training', alpha=1)
plt.hist(hist_data["before_up"], bins=10**np.linspace(-1, 2, 50), color=My_Orange, edgecolor='white', label='Before training', alpha=0.7)
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.xlabel(r"Upper bounds for relative errors (in $\%$)", labelpad=5)
plt.ylabel("Number of occurrences", labelpad=15)
plt.title("Upper-Bound Histogram")
plt.legend()
plt.gca().set_aspect(0.6)
plt.savefig(f"{RESULTS_FOLDER}/histogram_upper.png", bbox_inches='tight', dpi=300)
plt.show()

plt.hist(hist_data["after_low"], bins=10**np.linspace(-1, 2, 50), color='white', edgecolor=My_Blue, label='After training', alpha=1, linewidth=2)
plt.hist(hist_data["before_low"], bins=10**np.linspace(-1, 2, 50), color='white', edgecolor=My_Orange, label='Before training', alpha=0.7, linewidth=2)
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.xlabel(r"Lower bounds for relative errors (in $\%$)", labelpad=5)
plt.ylabel("Number of occurrences", labelpad=15)
plt.title("Lower-Bound Histogram")
plt.legend()
plt.gca().set_aspect(0.6)
plt.savefig(f"{RESULTS_FOLDER}/histogram_lower.png", bbox_inches='tight', dpi=300)
plt.show()

# Read loss data
loss_data = pd.read_csv(f"{RESULTS_FOLDER}/loss.csv", sep=";")
val_data = pd.read_csv(f"{RESULTS_FOLDER}/validation.csv", sep=";")

# Plot training and validation loss
plt.plot(loss_data["iteration"], loss_data["loss"], '-', color=My_Blue, linewidth=1.5, label='Training')
plt.plot(val_data["iteration"], val_data["loss_int"], 'o', color=My_Orange, linewidth=4, label='Int. Valid.', alpha=0.8)
plt.plot(val_data["iteration"], val_data["loss_test"], 'x', color=My_Red, linewidth=4, label='Test Valid.', alpha=0.8)
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.xlabel("Iteration", labelpad=5)
plt.ylabel(r"$\mathcal{L}_{DFR}(\alpha)$", labelpad=15)
plt.title("Training and validation history")
plt.legend()
plt.gca().set_aspect(0.6)
plt.savefig(f"{RESULTS_FOLDER}/loss.png", bbox_inches='tight', dpi=300)
plt.show()
