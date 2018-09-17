# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# import xlrd
# import matplotlib.pyplot as plt
# import os
# from sklearn.utils import check_random_state
#
#
# xdata=np.random.rand(100).astype(np.float32)
# ydata=3*xdata+2
#
# a=tf.Variable(1.0)
# b=tf.Variable(0.2)
#
# y=a*xdata+b
#
#
# loss=tf.reduce_mean(tf.square(y-ydata))
#
# optimizer=tf.train.GradientDescentOptimizer(0.5)
#
# train=optimizer.minimize(loss)
#
# init=tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     sess.run(init)
#     train_data=[]
#     for steps in range(100):
#         evals=sess.run([train,a,b])[1:]
#         if(steps%5==0):
#             print (steps,evals)
#             train_data.append(evals)


        #########################################################################################################
        ######################################## another one#####################################################
        ##########################################################################################################


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd
import matplotlib.pyplot as plt
import os
from sklearn.utils import check_random_state

# Generating artificial data.
n = 50
XX = np.arange(n)
rs = check_random_state(0)
YY = rs.randint(-20, 20, size=(n,)) + 2.0 * XX
data = np.stack([XX,YY], axis=1)


#######################
## Defining flags #####
#######################
tf.app.flags.DEFINE_integer(
    'num_epochs', 5, 'The number of epochs for training the model. Default=50')
# Store all elemnts in FLAG structure!
FLAGS = tf.app.flags.FLAGS


# creating the weight and bias.
# The defined variables will be initialized to zero.
W = tf.Variable(0.0, name="weights")
b = tf.Variable(0.0, name="bias")


#  Creating placeholders for input X and label Y.
def inputs():
    """
    Defining the place_holders.
    :return:
            Returning the data and label place holders.
    """
    X = tf.placeholder(tf.float32, name="X")
    Y = tf.placeholder(tf.float32, name="Y")
    return X,Y

# Create the prediction.
def inference(X):
    """
    Forward passing the X.
    :param X: Input.
    :return: X*W + b.
    """
    return X * W + b

def loss(X, Y):
    '''
    compute the loss by comparing the predicted value to the actual label.
    :param X: The input.
    :param Y: The label.
    :return: The loss over the samples.
    '''

    # Making the prediction.
    Y_predicted = inference(X)
    return tf.squared_difference(Y, Y_predicted)


# The training function.
def train(loss):
    learning_rate = 0.0001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


with tf.Session() as sess:


    sess.run(tf.global_variables_initializer())
    X, Y = inputs()
    # train_loss = loss(X, Y)

    for epoch_num in range(FLAGS.num_epochs): # run 100 epochs
        for x, y in data:

          train_op = train(loss(X,Y))
          loss_value,_ = sess.run([loss(X,Y),train_op], feed_dict={X: x, Y: y})

        print('epoch %d, loss=%f' %(epoch_num+1, loss_value))
        wcoeff, bias = sess.run([W, b])


###############################
#### Evaluate and plot ########
###############################
Input_values = data[:,0]
Labels = data[:,1]
Prediction_values = data[:,0] * wcoeff + bias

# # uncomment if plotting is desired!
plt.plot(Input_values, Labels, 'ro', label='main')
plt.plot(Input_values, Prediction_values, label='Predicted')

# Saving the result.
plt.legend()
plt.savefig('plot.png')
plt.close()
