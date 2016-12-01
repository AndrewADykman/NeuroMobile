import tensorflow as tf
import numpy as np
import pickle

"""GET DATA FROM ALEXNET PORTION -- ALEXNET outputs 13 x 13 x 256 and reshape"""

with open('CNN_filters.pickle','rb') as f:
    CNN_data = pickle.load(f)

with open('twist.pickle','r') as g:
    twist = pickle.load(g)

print "finished pickles"

num_batches = CNN_data.shape[0]
input_size = int(np.prod(CNN_data.shape[1:]))

twist = np.asarray(twist)

#define functions for initializing slightly positive random variables (TF_VARIABLES)
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#CREATE GRAPH #############

#define our placeholder variables for defining the symbolic expression to diff
x = tf.placeholder(tf.float32, shape=[None, input_size])
y_data = tf.placeholder(tf.float32, shape=[None, 2])

#run it through several FC dense layers
fc6_hidden_size = 1000
fc6W = weight_variable([input_size, fc6_hidden_size])
fc6b = bias_variable([fc6_hidden_size])

fc6 = tf.nn.relu(tf.matmul(x, fc6W) + fc6b)

#run it through several FC dense layers
fc7_hidden_size = 100
fc7W = weight_variable([fc6_hidden_size, fc7_hidden_size])
fc7b = bias_variable([fc7_hidden_size])

fc7 = tf.nn.relu(tf.matmul(fc6, fc7W) + fc7b)

#apply dropout
keep_prob = tf.placeholder(tf.float32)
fc7_drop = tf.nn.dropout(fc7, keep_prob)

#final readout layer (2 output nodes, dYaw and dx)
y_W = weight_variable([fc7_hidden_size, 2])
y_B = bias_variable([2])

y_pred = tf.matmul(fc7_drop, y_W) + y_B

#define loss function
squared_loss = tf.reduce_mean(tf.square(y_pred - y_data))

#define training step
train_step = tf.train.AdamOptimizer(1e-4).minimize(squared_loss)

#additional formulas for logging accuracy
correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_data,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#start the Tensorflow session and initialize variables
sess = tf.Session()
sess.run(tf.initialize_all_variables())

"""
this might not be the most efficient way to get batches. from Andrew:

numTrainingExamples = len(ppTD)
miniBatchSize = 10
miniBatchNums = []
miniBatch = []
while len(miniBatchNum) < miniBatchSize:
  num = np.random.randint(0, numTrainingExamples)
  if num not in miniBatchNums:
    miniBatchNums.append(num)
    miniBatch.append(ppTD[num])
"""

#for some number of iterations

miniBatchSize = 10

for i in range(20000):
    #draw random mini-batches, TODO still need to do sampling without replacement tho
    samples = np.random.randint(0, num_batches, miniBatchSize)
    print samples
    x_batch = CNN_data[samples]
    y_batch = twist[samples]

    #flatten x_batch
    x_batch = x_batch.flatten()

    #every 100 iterations print accuracy
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: x_batch, y_data: y_batch, keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))

    #train
    train_step.run(feed_dict={x: x_batch, y_data: y_batch, keep_prob: 0.5})



