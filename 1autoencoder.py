import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import math
import matplotlib.pyplot as plt

mnist=input_data.read_data_sets('./mnist',one_hot=True)

trainX=mnist.train.images[:2000]
trainY=mnist.train.labels[:2000]
n_filters=[1,10,10,10,10]
filter_sizes=[3,3,3,3]

#Input
inputdata=tf.placeholder(tf.float32,[None,784],name='in')
image=tf.reshape(inputdata,[-1,28,28,1])
outputdata=tf.placeholder(tf.float32,[None,10],name='out')

def encoder():
	#first layer
	conv1=tf.layers.conv2d(inputs=image,filters=16,kernel_size=5,strides=1,
							padding='same',activation=relu)


# plt.imshow(trainX[1].reshape(28,28),cmap="Greys")
# plt.show()
learning_rate=0.01


# def autoencoder()













