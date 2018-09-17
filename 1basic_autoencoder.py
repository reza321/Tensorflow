import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt

images=np.loadtxt('fashionmnist/fashion-mnist_train.csv',delimiter=',',skiprows=1)[:,1:]
# print(images.shape)
# print(images[0])
# plt.imshow(images[1].reshape(28,28),cmap="Greys")
# plt.show()

input_dim=784
hidd1_dim=32

hidd2_dim=32
output_dim=784

#first hidden layer 784*32 weights and 32 biases
hidd1_val={'weights':tf.Variable(tf.random_normal([input_dim,hidd1_dim])),
           'biases':tf.Variable(tf.random_normal([hidd1_dim]))}
hidd2_val={'weights':tf.Variable(tf.random_normal([hidd1_dim,hidd2_dim])),
            'biases':tf.Variable(tf.random_normal([hidd2_dim]))}

output_val={'weights':tf.Variable(tf.random_normal([hidd2_dim,output_dim])),
            'biases':tf.Variable(tf.random_normal([output_dim]))}

#define neural net.
input_data=tf.placeholder('float',[None,input_dim])

layer1=tf.nn.sigmoid(tf.add(tf.matmul(input_data,hidd1_val['weights']),
                        hidd1_val['biases']))
layer2=tf.nn.sigmoid(tf.add(tf.matmul(layer1,hidd2_val['weights']),
                        hidd2_val['biases']))

outputlayer=tf.nn.sigmoid(tf.add(tf.matmul(layer2,output_val['weights']),
                        output_val['biases']))
output_data=tf.placeholder('float',[None,input_dim])

# CostFunction
meansq=tf.reduce_mean(tf.square(outputlayer-output_data))
# Optimizer
learning_rate=0.1
optimizer=tf.train.AdagradOptimizer(learning_rate).minimize(meansq)

init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

# BATCH size
batch_size=100
hm_epochs=100
tot_images=60000

for epoch in range(hm_epochs):
    epoch_loss=0
    for i in range(int(tot_images/batch_size)):
        epoch_x=images[i*batch_size:(i+1)*batch_size]

        _,c=sess.run([optimizer,meansq],
            feed_dict={input_data:epoch_x,output_data:epoch_x})
        epoch_loss+=c
    print(epoch_loss)


anyimage=images[999]
plt.imshow(anyimage.reshape(28,28),cmap="Greys")
plt.show()

output_anyimage=sess.run(outputlayer,feed_dict={input_data:[anyimage]})

plt.imshow(output_anyimage.reshape(28,28),  cmap='Greys')
plt.show()








































