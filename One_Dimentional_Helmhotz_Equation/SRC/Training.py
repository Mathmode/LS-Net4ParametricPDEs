
import tensorflow as tf

dtype="float64"
tf.keras.backend.set_floatx(
    dtype
)
from SRC.Loss1D import make_loss
from SRC.LSArchitecture import update_u



# Define optimizer with decreasing learning rate
def lr_schedule(epoch, initial_learning_rate, final_learning_rate, num_epochs):
    decay_rate = (final_learning_rate / initial_learning_rate) ** (1 / num_epochs)
    current_learning_rate = initial_learning_rate * (decay_rate ** epoch)
    return current_learning_rate

###############################################################################
###############################################################################
###############################################################################

@tf.function
def train_step(u_model, u_bases, P_data, x, DST, DCT, DCT_a, DCT_b, norms, nvar, optimizer):
    with tf.GradientTape() as tape:
        loss_tot = tf.constant([0],dtype=dtype) 
        num_tot = 0
        
        for p in P_data:
            cond = update_u(u_model, u_bases, p, nvar, x, DST, DCT, DCT_a, DCT_b, norms)
            loss = make_loss(u_model, p, x, DST, DCT, DCT_a, DCT_b, norms)
            loss_tot+=loss**0.5
            num_tot +=1
    
        Loss = tf.squeeze((loss_tot)/tf.cast(num_tot,dtype))
    gradients = tape.gradient(Loss, u_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, u_model.trainable_variables))
    
    return Loss

###############################################################################
###############################################################################
###############################################################################

@tf.function
def loss_evaluation(u_model, u_bases, P_data, x, DST, DCT, DCT_a, DCT_b, norms, nvar, optimizer):

    loss_tot = tf.constant([0],dtype=dtype) 
    num_tot = 0
        
    for p in P_data:
        cond = update_u(u_model, u_bases, p, nvar, x, DST, DCT, DCT_a, DCT_b, norms)
        loss = make_loss(u_model, p, x, DST, DCT, DCT_a, DCT_b, norms)
        loss_tot+=loss**0.5
        num_tot +=1
          
    Loss = tf.squeeze((loss_tot)/tf.cast(num_tot,dtype))
    return Loss