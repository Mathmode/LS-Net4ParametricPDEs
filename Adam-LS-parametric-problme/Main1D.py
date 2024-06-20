# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 2024

@author: Shima Baharlouei
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time as time

from SRC.Architecture1D import make_u_model
from SRC.Test_functions import test_functions
from SRC.Loss1D import make_loss, indicator, sigma
from SRC.LSArchitecture import update_u
from SRC.ProblemHelmholtz1D import u_exact, rhs
from SRC.Training import train_step, lr_schedule, loss_evaluation

dtype="float64"
tf.keras.backend.set_floatx(
    dtype
)

###############################################################################
###############################################################################
###############################################################################
#Data

neurons = 20
neurons_final= 36
nvar = neurons_final

npts=1000
nmodes = 400
max_freq=1.5
min_freq = -1.5

dat_len=500
dat_dim=4
batch_size=1

num_epochs = 20 # Number of epochs
initial_learning_rate = 0.01
final_learning_rate = 0.001


###############################################################################
###############################################################################
###############################################################################
# validation parameter set define randomly in each epoch
Xi_val = 10**(tf.random.uniform([int(dat_len/10+1),dat_dim],dtype=dtype,minval=min_freq,maxval=max_freq))

x, DST, DCT, norms = test_functions(nmodes, npts)
x_val, DST_val, DCT_val, norms_val = test_functions(nmodes, npts + 5)

u_model, u_bases = make_u_model(neurons,neurons_final=neurons_final,activation=tf.math.sin)
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
u_model.compile(optimizer=optimizer,loss = make_loss)

# Training loop
Loss = []
Loss_val = []
for epoch in range(num_epochs):
    # training parameter set define randomly in each epoch
    Xi_data = 10**(tf.random.uniform([dat_len,dat_dim],dtype=dtype,minval=min_freq,maxval=max_freq))
    #Changing the learning rate
    new_learning_rate = lr_schedule(epoch, initial_learning_rate, final_learning_rate, num_epochs)
    optimizer.learning_rate.assign(new_learning_rate)
    
    Loss += [train_step(u_model, u_bases, Xi_data, x, DST, DCT, norms, nvar, optimizer)]
    Loss_val += [loss_evaluation(u_model, u_bases, Xi_val, x_val, DST_val, DCT_val, \
                                 norms_val, nvar, optimizer)]
    print('=====================================')
    print('=============', 'Epoch:', epoch, '/', num_epochs,'=============')
    print('Loss_training:', float(Loss[-1]))
    print('Loss_validation:', float(Loss_val[-1]))


###############################################################################
###############################################################################
###############################################################################
###############################################################################################################
      
        
        
font = {'family' : 'Times',
        'weight' : 'normal',
        'size'   : 23}
plt.rc('font', **font)
plt.figure(figsize=(6, 4))
plt.rcParams['text.usetex'] = True
plt.plot([(i) for i in range(1,num_epochs+1)],(np.array(Loss)),'-r', linewidth=1, label = 'Training')

plt.plot([(i) for i in range(1,num_epochs+1)],(np.array(Loss_val)),'-b', linewidth=1, label =  'Validation')
#plt.title('PINN')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.xscale("log")
plt.yscale("log")
#plt.xticks([1, 10, 100, 1000], ['$10^0$', '$10^1$', '$10^2$', '$10^3$'])
plt.subplots_adjust(left  =0.3,
                    bottom=0.19,
                    right =0.95,
                    top   =0.8,
                    wspace=0.1,
                    hspace=0.2)
plt.savefig('Loss_0.pdf',dpi=300)
plt.show()
###############################################################################
###############################################################################
###############################################################################

#test

xtest = tf.constant([((i+0.5)/1000) for i in range(1000)],dtype=dtype)
    
for i in range(1):
    Atest = 10**(tf.random.uniform([dat_dim],dtype=dtype,minval=min_freq,maxval=max_freq))
    # Atest = tf.constant([1.,1.1,1.],dtype=dtype)
    cond = update_u(u_model, u_bases, Atest, nvar, x, DST, DCT, norms)
    
    with tf.GradientTape(persistent=True) as t1:
        t1.watch(xtest)
        u = u_model(xtest)
        ue = u_exact(xtest,Atest[0],Atest[1],Atest[2],Atest[3])
    
    due = t1.gradient(ue,xtest)
    du = t1.gradient(u,xtest)
    
    font = {'family' : 'Times',
            'weight' : 'normal',
            'size'   : 23}
    plt.rc('font', **font)
    plt.figure(figsize=(6, 4))
    plt.rcParams['text.usetex'] = True
    
    plt.plot(xtest,u, label = 'u_test')
    plt.plot(xtest,ue, label = 'u_exact')
    plt.subplots_adjust(left  =0.3,
                        bottom=0.19,
                        right =0.95,
                        top   =0.8,
                        wspace=0.1,
                        hspace=0.2)
    plt.savefig('test0.pdf',dpi=300)
    plt.show()
    # plt.plot(xtest,sigma(xtest,Atest[0],Atest[0],Atest[0])*du, label = 'du_test')
    # plt.plot(xtest,sigma(xtest,Atest[0],Atest[0],Atest[0])*due, label = 'du_exact')
    # plt.legend()
    # plt.show()
print("L2 err = ", float(tf.reduce_sum((ue-u)**2)/tf.reduce_sum(ue**2))**0.5*100)
print("H1 err = ", float(tf.reduce_sum((due-du)**2)/tf.reduce_sum(due**2))**0.5*100)
    
    
