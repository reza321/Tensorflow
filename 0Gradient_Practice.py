# import numpy as np
# import tensorflow as tf

# xdata=np.random.rand(1).astype(np.float32)
# ydata=3*xdata+2

# print(xdata)
# w=tf.Variable(1.0)
# b=tf.Variable(0.2)
# y=w*xdata+b


# learning_rate=0.1
# loss=tf.reduce_mean(tf.square(y-ydata))

# train=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# dw,db=tf.gradients(loss,[w,b])

# init=tf.global_variables_initializer()

# with tf.Session() as sess:
# 	sess.run(init)
# 	train_data=[]
# 	for steps in range(10):
# 		evals=sess.run([train,w,b,dw,db])
# 		if (steps%5==0):
# 			print(steps,evals)

# ---------------------- EXAMPLE 2 ---------------------------------
from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

learning_rate=0.01
training_epochs=5
batch_size=100
display_step=1

x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])

W=tf.Variable(tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))

pred= tf.nn.softmax(tf.matmul(x,W)+b)
cost=tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred),reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

dW,db=tf.gradients(cost,[W,b])
init=tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	for epoch in range(training_epochs):
		avg_cost=0
		total_batch=int(mnist.train.num_examples/batch_size)
		for i in range(total_batch):
			batch_x,batch_y=mnist.train.next_batch(batch_size)
			[deltaW,deltab,c]=sess.run([dW,db,cost],feed_dict={x:batch_x,y:batch_y})
			print(deltaW.shape)
			print(deltab.shape)
			exit()
			avg_cost+=c/total_batch
		if ((epoch+1) % display_step) == 0:
			print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
			print(deltaW)

	print("Optimization Finished!")

	# Test model
	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	# Calculate accuracy
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))



















