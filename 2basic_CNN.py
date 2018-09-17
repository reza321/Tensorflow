
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt
import numpy as np

def weight_variable(shape):
    initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.random_normal(shape, mean=0.0, stddev=0.01)
    return tf.Variable(initial)


def montage(W):
    W = np.reshape(W, [W.shape[0], W.shape[1], 1, W.shape[2] * W.shape[3]])
    n_plots = int(np.ceil(np.sqrt(W.shape[-1])))
    m = np.ones(
        (W.shape[0] * n_plots + n_plots + 1,
         W.shape[1] * n_plots + n_plots + 1)) * 0.5
    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < W.shape[-1]:
                m[1 + i + i * W.shape[0]:1 + i + (i + 1) * W.shape[0],
                  1 + j + j * W.shape[1]:1 + j + (j + 1) * W.shape[1]] = (
                    np.squeeze(W[:, :, :, this_filter]))
    return m




mnist=input_data.read_data_sets('MNIST_data/',one_hot=True)

x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])


x_tensor=tf.reshape(x,[-1,28,28,1])


filter_size=5

n_filter1=16
w_conv1=weight_variable([filter_size,filter_size,1,n_filter1])
b_conv1=bias_variable([n_filter1])

h_conv1=tf.nn.relu(tf.nn.conv2d(input=x_tensor,filter=w_conv1,
								strides=[1,2,2,1],padding='SAME')+b_conv1)

n_filter2=16
w_conv2=weight_variable([filter_size,filter_size,n_filter1,n_filter2])
b_conv2=bias_variable([n_filter2])

h_conv2=tf.nn.relu(tf.nn.conv2d(input=h_conv1,filter=w_conv2,
								strides=[1,2,2,1],padding='SAME')+b_conv2)


h_conv2flat=tf.reshape(h_conv2,[-1,7*7*n_filter2])


# Create a fully-connected layer
n_fc=1024
w_fc1=weight_variable([7*7*n_filter2,n_fc])
b_fc1=bias_variable([n_fc])
h_fc1=tf.nn.relu(tf.matmul(h_conv2flat,w_fc1)+b_fc1)

# We can add dropout for reg
keep_prob=tf.placeholder(tf.float32)
h_fc1drop=tf.nn.dropout(h_fc1,keep_prob)


# And finally our softmax layer:
w_fc2=weight_variable([n_fc,10])
b_fc2=bias_variable([10])
y_pred=tf.nn.softmax(tf.matmul(h_fc1drop,w_fc2)+b_fc2)


# Define loss/eval
cross_energy=-tf.reduce_sum(y*tf.log(y_pred))
optimizer=tf.train.AdamOptimizer().minimize(cross_energy)


# Monitor accuracy
correct_prediction=tf.equal(tf.argmax(y_pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

sess=tf.Session()
sess.run(tf.global_variables_initializer())


# minibatch
batch_size=100
epochs=1
for epoch in range(epochs):
  for batchi in range(mnist.train.num_examples//batch_size):
    batch_x,batch_y=mnist.train.next_batch(batch_size)
    # flat=sess.run(tf.shape(h_conv2flat),feed_dict={x:batch_x,y:batch_y,keep_prob:0.5})
    # print(flat)
    sess.run(optimizer,feed_dict={x:batch_x,y:batch_y,keep_prob:0.5})    
    print(sess.run(accuracy,
                   feed_dict={
                       x: mnist.validation.images,
                       y: mnist.validation.labels,
                       keep_prob: 1.0
                   }))


# %% Let's take a look at the kernels we've learned
W = sess.run(w_conv1)
plt.imshow(montage(W / np.max(W)), cmap='coolwarm')



