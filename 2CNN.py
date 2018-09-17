##################           https://www.tensorflow.org/tutorials/estimators/cnn
import tensorflow as tf
import numpy as np
import keras


def cnn_model_fn(inputdata,labels):
    input_layer=tf.reshape(inputdata,[-1,28,28,1]) # shape: [batch_size,height,width,channels]
                                                   # ch=1: Black, ch=3:RGB
    conv1=tf.layers.conv2d(inputs=input_layer,filters=32,kernel_size=[5,5],padding='same',activation=tf.nn.relu,name="conv1")
                        # we have 32 filters with the size of (5*5)
                        # padding=same ---> zero padding ---> output-size=input-size 
                        #       => output=[batchsize,28,28,32]


    pool1=tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],strides=2,name="pool1")
                                 # pool_size[2,2]: from input of 28*28 it picks largest value
                                 # of elemnts every each other 
                                 #strides: jumps of pool filter --> output: 14*14
                                 #       => output=[batchsize,14,14,32]

    conv2=tf.layers.conv2d(inputs=pool1,filters=64,kernel_size=5,padding='same',
                            activation=tf.nn.relu,name="conv2")                                 
                                 #       => output=[batchsize,14,14,32]    
    pool2=tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],strides=[2,1],name="pool2")
                                  # Output 7*7
                                 #       => output=[batchsize,7,13 not 14!,64]
    #Dense_layer
    pool2_flat=tf.reshape(pool2,[-1,7*13*64])    #  => output=[batchsize,7*7*64=3136]
    dense = tf.layers.flatten(pool2_flat)
    dense=tf.layers.dense(inputs=dense,units=1024,activation=tf.nn.relu,name="dense")
    dropout=tf.layers.dropout(inputs=dense,rate=0.5) # => output=[batchsize,1024]
    logits = tf.layers.dense(inputs=dropout,units= 10,name="logits") # => output=[batchsize,10]      
    predicted=tf.argmax(input=logits,axis=1)    # Gives classes
    probabilities=tf.nn.softmax(logits,name="softmax_tensor") # Gives probabilities of classes

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)    

    # conv1_kernel_val = tf.get_default_graph().get_tensor_by_name('dense/kernel:0')
    # conv1_bias_val = tf.get_default_graph().get_tensor_by_name('logits/bias:0')
    # return loss,optimizer,tf.shape(pool2),conv1_bias_val,conv1_kernel_val
    all_params = []
    for layer in ['conv1', 'conv2', 'dense', 'logits']:        
        for var_name in [ 'kernel','bias']:
            temp_tensor = tf.get_default_graph().get_tensor_by_name("%s/%s:0" % (layer, var_name))            
            all_params.append(temp_tensor)          
    return loss,optimizer,tf.shape(pool2),all_params





mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
test_data = mnist.test.images # Returns np.array
test_labels = np.asarray(mnist.test.labels, dtype=np.int32)



# minibatch
batch_size=100
epochs=5
with tf.Session() as sess:
  input_labeled = tf.placeholder(dtype=tf.float32, shape=(None,784), name='input_labeled')
  true_label = tf.placeholder(tf.int32, shape=(None,), name='true_label')
  model=cnn_model_fn(input_labeled,true_label)
  sess.run(tf.global_variables_initializer())  
  batch_x,batch_y=mnist.train.next_batch(100)
  

  # op = sess.graph.get_operations()
  # print([m.name for m in op])
  # print([m.values() for m in op][1])
  # exit()
  
  for epoch in range(epochs):
    for batchi in range(10):
      batch_x,batch_y=mnist.train.next_batch(batch_size)
      loss,_,pool2shape,params=sess.run(model,feed_dict={input_labeled:batch_x,true_label:batch_y})    
      print(loss) # If optimizer does not get called, then loss does not change.
      print(params)



