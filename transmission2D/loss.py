#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Last edited on Sept, 2024

@author: curiarteb
"""

from config import Aparam,Bparam,P,M1,M2,MVAL1,MVAL2,K1,K2,KVAL1,KVAL2,SOURCE,LIFT,DLIFTX,LEARNING_RATE,DTYPE
from integration import integration_points_and_weights
from models import test_functions
import tensorflow as tf
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
keras.config.set_dtype_policy(DTYPE)
keras.mixed_precision.set_global_policy(DTYPE)

CENTERS = tf.expand_dims(tf.constant([
    [0.5, 0.5],   # Center of Omega1
    [-0.5, 0.5],  # Center of Omega2
    [0.5, -0.5],  # Center of Omega3
    [-0.5, -0.5]  # Center of Omega4
], dtype=DTYPE), axis=0)  # [1, 4, 2]

def sigma_values(x, y, sigma):
    """
    Function to evaluate sigma(x, y; sigma1, sigma2, sigma3, sigma4)

    x: Tensor of shape [K, 1] (batch of x-coordinates)
    y: Tensor of shape [K, 1] (batch of y-coordinates)
    S: Tensor of shape [P, 4] representing the samples of [sigma1, ..., sigma4]

    Returns:
    Tensor of shape [P, K] where each element is the output of sigma(...)
    for the corresponding combination of (x, y) and row in S.
    """

    # Stack x and y for easier broadcasting: shape will be [K, 2]
    xy = tf.concat([x, y], axis=-1)  # [K, 2]

    # Expand dimensions to broadcast over both K and P
    xy_expanded = tf.expand_dims(xy, axis=1)  # [K, 1, 2]
    
    # Calculate squared distances: shape [K, 4]
    distances_squared = tf.reduce_sum((xy_expanded - CENTERS) ** 2, axis=-1)  # [K, 4]
    
    # Determine if (x, y) is in each circle: shape [K, 4]
    in_circle = distances_squared <= (1/4) ** 2  # [K, 4]

    # Expand dimensions to broadcast over both N and M
    in_circle_expanded = tf.expand_dims(in_circle, axis=1)  # [K, 1, 4]
    D_expanded = tf.expand_dims(sigma, axis=0)  # [1, P, 4]

    # Multiply masks with D, sum over circles
    values = tf.reduce_sum(tf.where(in_circle_expanded, D_expanded, tf.zeros_like(D_expanded)), axis=-1)  # [P, K]

    # If no circle contains the point, return 1
    return tf.where(tf.reduce_any(in_circle, axis=1, keepdims=True), values, tf.ones_like(values))

class loss_individual(keras.Model):
    
    def __init__(self, net, label="train",**kwargs):
        super(loss_individual, self).__init__()
        
        self.sigma = sigma_values
        self.f = SOURCE

        self.net = net   
        self.label = label
        
        if self.label == "train":
            self.test = test_functions(spectrum_size=[M1,M2])
            self.data = integration_points_and_weights(threshold=[K1,K2])
        elif self.label == "intval":
            self.test = test_functions(spectrum_size=[M1,M2])
            self.data = integration_points_and_weights(threshold=[KVAL1,KVAL2])
        elif self.label == "testval":
            self.test = test_functions(spectrum_size=[MVAL1,MVAL2])
            self.data = integration_points_and_weights(threshold=[KVAL1,KVAL2])

    def build(self, input_shape):
        super(loss_individual, self).build(input_shape)
    
    def construct_LHMandRHV(self, sigma): #[P,4]
        
        x,y,w = self.data() #[K,1],[K,1],[K,1]
        inputs = [x,y]

        # Test functions
        v = self.test(inputs) # [K,M]
        dvX, dvY = self.test.gradient(inputs) # [K,M],[K,M]
        
        # Source
        f = self.f(x,y) # [K,1]
        wfv = tf.einsum("kr,kr,km->mr", w, f, v) # [M,1]
        wfv = tf.expand_dims(wfv, axis=0) # [1,M,1] for broadcasting
        
        # Sigma coeff
        sigma_coeff = self.sigma(x,y,sigma) # [K,P]
        
        # Lift
        dliftX = DLIFTX(x,y)
        #lift = LIFT(x,y)
        sigma_laplacian_lift = tf.einsum("kr,kp,kr,km->pmr", w, sigma_coeff, dliftX, dvX) #[P,M,1]
        #reaction_lift = tf.einsum("kr,kr,km->mr", w, lift, v) #[M,1]
        #reaction_lift = tf.expand_dims(reaction_lift, axis=0) #[1,M,1] for broadcasting
        
        # Right-hand side vector construction
        self.l = wfv - sigma_laplacian_lift #+ reaction_lift #[P,M,1]
        
        # Left-hand side (bilinear-form matrix)
        duX, duY = self.net.dfwd_vect(inputs) #[K,N],[K,N]
        wduXdvX = tf.einsum("kr,kp,kn,km->pmn", w, sigma_coeff, duX, dvX) #[P,M,N] 
        wduYdvY = tf.einsum("kr,kp,kn,km->pmn", w, sigma_coeff, duY, dvY) #[P,M,N]
        sigma_laplacian = wduXdvX + wduYdvY #[P,M,N]

        #u = self.net.call_vect(inputs) #[K,N]
        #reaction = tf.einsum("kr,kn,km->mn", w, u, v) #[M,N]
        #reaction = tf.expand_dims(reaction, axis=0) #[1,M,N] for broadcasting
       
        self.B = sigma_laplacian #- (math.pi/2)**2*reaction #[P,M,N]
    
    
    # Solve the LS system of linear equations with l2 regularization
    def optimal_computable_vars(self, regularizer = 10**(-3)):
        
        self.weights_optimal = tf.linalg.lstsq(self.B, self.l, l2_regularizer=regularizer) # [P,N,1]

        
    # Compute || B w - l ||^2 given B, w and l
    def from_LHMandRHV_to_loss(self):
        residual = tf.einsum("pmn,pnr->pmr", self.B, self.weights_optimal) - self.l
        loss = tf.reduce_sum(residual**2, axis=1)
    
        return loss
    
    def call(self, sigma):
        
        self.construct_LHMandRHV(sigma)
        self.optimal_computable_vars()
        loss = self.from_LHMandRHV_to_loss()
        
        return loss
    
    def MC(self, sigma):
        
        loss = self(sigma)
        return tf.reduce_mean(loss)
    
    
class training(keras.Model):
    
    def __init__(self, net, **kwargs):
        super(training, self).__init__()

        self.net = net
        self.trainable_vars = [v.value for v in self.net.weights[:-1]]
        self.computable_vars = self.net.weights[-1].value
        
        self.loss_train = loss_individual(net, label="train")
        self.loss_intval = loss_individual(net, label="intval")
        self.loss_testval = loss_individual(net, label="testval")

        # Weights for the LS computations for 'net'
        self.computable_vars = self.net.weights[-1].value
        

    def compile(self):
        super().compile()
        
        self.optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    def train_step(self, data):
        
        sigma = tf.random.uniform(shape=[P,4],minval=Aparam,maxval=Bparam,dtype=DTYPE)
        # loss_intval = self.loss_intval.MC(sigma)
        # loss_testval = self.loss_testval.MC(sigma)
        
        # Optimize net
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(self.trainable_vars)
            loss_train = self.loss_train.MC(sigma)
        gradient = tape.gradient(loss_train, self.trainable_vars)
        self.optimizer.apply(gradient, self.trainable_vars)
        
        return {"loss_train": loss_train}#, "loss_intval": loss_intval, "loss_testval": loss_testval}
    
    
class u_lifted(keras.Model):
    def __init__(self, net, **kwargs):
        super(u_lifted, self).__init__()
        
        # Hidden layers
        self.net = net
        self.lift = LIFT
        self.loss = loss_individual(self.net,label="testval")
        
    def build(self, input_shape):
        super(u_lifted, self).build(input_shape)
    
    # Returns the scalar output of the neural network
    def call(self, inputs):
        
        x,y,sigma=inputs
        self.loss(sigma)
        self.net.weights[-1].assign(tf.reshape(self.loss.weights_optimal,[-1,1]))
        
        return self.net([x,y]) + self.lift(x,y)
    
class relative_error_lower_and_upper(keras.Model):
    def __init__(self, net, **kwargs):
        super(relative_error_lower_and_upper, self).__init__()
        
        self.net = net
        self.loss = loss_individual(net,label="testval")
        
    def call(self, sigma):
        
        loss = self.loss(sigma) #[P,1],[P,1]
        optimal_coeff = tf.squeeze(self.loss.weights_optimal) #[P,N]
        
        x,y,w = self.loss.data()
        duX, duY = self.net.dfwd_vect([x,y]) #[K,N],[K,N]
        duXp2 = tf.square(tf.einsum("kn,pn->kp", duX, optimal_coeff)) #[K,P]
        duYp2 = tf.square(tf.einsum("kn,pn->kp", duY, optimal_coeff)) #[K,P]
        wduXp2 = tf.einsum("kr,kp->pr", w, duXp2) #[P,1]
        wduYp2 = tf.einsum("kr,kp->pr", w, duYp2) #[P,1]
        norm = tf.sqrt(wduXp2 + wduYp2) #[P,1]
        
        sigma_max = tf.reduce_max(sigma, axis=1, keepdims=True)        
        sigma_min = tf.ones_like(sigma_max) #tf.reduce_min(sigma,axis=1,keepdims=True) #[P,1]

        
        estimate_upper = (1/sigma_min * tf.sqrt(loss)) / (norm - 1/sigma_min * tf.sqrt(loss)) * 100 #[P,1]
        estimate_lower = (1/sigma_max * tf.sqrt(loss)) / (norm + 1/sigma_max * tf.sqrt(loss)) * 100
        
        return estimate_lower, estimate_upper
        
        
