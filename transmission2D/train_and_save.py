#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited on Sept, 2024

@author: curiarteb
"""

import tensorflow as tf
import pandas as pd
import os
from models import u_net
from loss import training, u_lifted, relative_error_lower_and_upper
from callbacks import Validation
from config import Aparam, Bparam, RESULTS_FOLDER, EPOCHS, XYPLOT, DTYPE

Aparam, Bparam = int(Aparam), int(Bparam)

import matplotlib.pyplot as plt
font = {'weight': 'normal',
        'size': 15}
plt.rc('font', **font)

# Create the results folder if it does not exist
if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)

x0, y0 = XYPLOT

# For the Cartesian product
x = tf.repeat(x0, y0.shape[0], axis=0)
y = tf.tile(y0, (x0.shape[0], 1))

net = u_net()
net([x, y])

# For visual check
sigma_materials = tf.constant([[Bparam, Aparam, Aparam, Bparam]], dtype=DTYPE)
sigma_constant = Aparam * tf.ones(shape=[1, 4], dtype=DTYPE)
net_lifted = u_lifted(net)

# For histogram
sigma_hist = tf.random.uniform(shape=[500, 4], minval=Aparam, maxval=Bparam, dtype=DTYPE)

print()
print("#######################")
print(" PLOTS BEFORE TRAINING ")
print("#######################")
print()

# Two network predictions
pred_constant = net_lifted([x, y, sigma_constant])
p = plt.scatter(x, y, c=pred_constant, cmap="plasma", alpha=0.7)
plt.colorbar(p)
plt.xlabel("x", labelpad=5)
plt.ylabel("y", labelpad=15)
plt.title(f"Network prediction for p=({Aparam},{Aparam},{Aparam},{Aparam})")
plt.show()

pred_materials = net_lifted([x, y, sigma_materials])
p = plt.scatter(x, y, c=pred_materials, cmap="plasma", alpha=0.7)
plt.colorbar(p)
plt.xlabel("x", labelpad=5)
plt.ylabel("y", labelpad=15)
plt.title(f"Network prediction for p=({Bparam},{Aparam},{Aparam},{Bparam})")
plt.show()

print()
print("########################")
print("        TRAINING        ")
print("########################")
print()
train = training(net)
train.compile()
error = relative_error_lower_and_upper(net)
data_hist = train.loss_testval.data() + [sigma_hist]

before_hist_lower, before_hist_upper = error(sigma_hist)
before_hist_lower, before_hist_upper = tf.squeeze(before_hist_lower), tf.squeeze(before_hist_upper)
call = Validation(sigma_hist,num_of_validations=30)
history = train.fit(sigma_constant, epochs=EPOCHS, callbacks=[call])
after_hist_lower, after_hist_upper = error(sigma_hist)
after_hist_lower, after_hist_upper = tf.squeeze(after_hist_lower), tf.squeeze(after_hist_upper)

print()
print("########################")
print("       SAVING DATA      ")
print("########################")
print()

# Two network predictions
# With constant p
pred_constant = net_lifted([x, y, sigma_constant])
df_pred_constant = pd.DataFrame(tf.concat([x, y, pred_constant], axis=1), columns=["x", "y", "pred"])
df_pred_constant.to_csv(f"{RESULTS_FOLDER}/pred_constant.csv", sep=";", index=False)

# With pieceise-constant p
pred_materials = net_lifted([x, y, sigma_materials])
df_pred_materials = pd.DataFrame(tf.concat([x, y, pred_materials], axis=1), columns=["x", "y", "pred"])
df_pred_materials.to_csv(f"{RESULTS_FOLDER}/pred_materials.csv", sep=";", index=False)

# Histograms
df_hist = pd.DataFrame(tf.stack([before_hist_lower, before_hist_upper, after_hist_lower, after_hist_upper], axis=1).numpy(), columns=["before_low", "before_up", "after_low", "after_up"])
df_hist.to_csv(f"{RESULTS_FOLDER}/histogram.csv", sep=";", index=False)

# Training and validation loss
rang = tf.range(1, EPOCHS + 1, dtype=DTYPE)
loss_hist = tf.convert_to_tensor(history.history["loss_train"], dtype=DTYPE)
rang_val = tf.convert_to_tensor(history.history["iteration_val"], dtype=DTYPE)
loss_intval_hist = tf.convert_to_tensor(history.history["loss_intval"], dtype=DTYPE)
loss_testval_hist = tf.convert_to_tensor(history.history["loss_testval"], dtype=DTYPE)

df_loss = pd.DataFrame(tf.stack([rang, loss_hist], axis=1),
                       columns=["iteration", "loss"])
df_loss["iteration"] = df_loss["iteration"].astype(int)
df_loss.to_csv(f"{RESULTS_FOLDER}/loss.csv", sep=";", index=False)

df_val = pd.DataFrame(tf.stack([rang_val, loss_intval_hist, loss_testval_hist], axis=1),
                       columns=["iteration", "loss_int", "loss_test"])
df_val["iteration"] = df_val["iteration"].astype(int)
df_val.to_csv(f"{RESULTS_FOLDER}/validation.csv", sep=";", index=False)
