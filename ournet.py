import tensorflow as tf
import numpy as np
from data_classes import AlexTrData, RawTrData

"""GET DATA FROM ALEXNET PORTION -- ALEXNET outputs 13 x 13 x 256 """

hidden_size = 1000

#initialize slightly positive random variables
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#run it through several FC dense layers
fc6_hidden_size = 1000
fc6_input_size = 13*13*256
fc6W = weight_variable([fc6_input_size, fc6_hidden_size])
fc6b = bias_variable([fc6_hidden_size])

#what if we reshape before inputting
hpool = tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))])
fc6 = tf.nn.relu(tf.matmul(hpool, fc6W) + fc6b)

#run it through several FC dense layers
fc7_hidden_size = 100
fc7W = weight_variable([fc6_hidden_size, fc7_hidden_size])
fc7b = bias_variable([fc7_hidden_size])

fc7 = tf.nn.relu(tf.matmul(fc6, fc7W) + fc7b)

#apply dropout
keep_prob = tf.placeholder(tf.float32)
fc7_drop = tf.nn.dropout(fc7, keep_prob)

#our custom activation function that outputs it some desirable range, some sigmoid probs
def sigma(x):
    return tf.div(tf.constant(1.0),
                  tf.add(tf.constant(1.0), tf.exp(tf.neg(x))))

#time to train!
#get trainingData, a list of data as RawTrData
ppTD = []
#to push thru CNN, stop it from being for'd
for inst in trainingData:
  ppTD.append(inst.getAlexTrData())
  #preprocesses all of the sets thru the AlexCNN

#training:
##pulls mini-batches randomly (with replacement)
numTrainingExamples = len(ppTD)
miniBatchSize = 10
miniBatchNums = []
miniBatch = []
while len(miniBatchNum) < miniBatchSize:
  num = np.random.randint(0, numTrainingExamples)
  if num not in miniBatchNums:
    miniBatchNums.append(num)
    miniBatch.append(ppTD[num])
##compute differences
##compute gradient
##descend
##GOTO start



#this is tutorial code from tensorflow.com. we will want to change this to do least squares regression (or logistic) instead
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


