import tensorflow as tf
import numpy as np
import pickle
import argparse
import atexit

# DEFINE HYPERPARAMETERS
parser = argparse.ArgumentParser(description='take training arguments.')
parser.add_argument('--num-epochs', type=int, default=300, help='The number of epochs')
parser.add_argument('--keep-prob', type=float, default=.75, help='Probability of an edge being kept in the network each training iteration')
parser.add_argument('--seq-size', type=int, default=30, help='# of images of a sequence')
args = parser.parse_args()

@atexit.register
def saveData():

	print 'saving data...'

	np.save('error_rates.npy', error_rates)

	dnn_net_data = {}
	dnn_net_data['fc6'] = []
	dnn_net_data['fc6'].append(fc6W.eval(sess))
	dnn_net_data['fc6'].append(fc6b.eval(sess))

	dnn_net_data['y'] = []
	dnn_net_data['y'].append(y_W.eval(sess))
	dnn_net_data['y'].append(y_B.eval(sess))

	np.save('dnn_net_data.npy', dnn_net_data)

#keep batch size small
seq_size = args.seq_size

num_epochs = args.num_epochs

keep_probability = args.keep_prob

#9 classes of output
output_num = 9

# OPEN PICKLED OUTPUT FROM ALEXNET
with open('CNN_filters_train.pickle','rb') as f:
    CNN_data = pickle.load(f)

with open('train_twists.pickle','r') as g:
    twist = pickle.load(g)

print "finished pickles"

#SMALL SUBSET OF DATA TEST
#CNN_data = CNN_data[50:71]
#twist = twist[50:71]

#get some dimensions
CNN_data = np.asarray(CNN_data)
num_batches = CNN_data.shape[0]
image_dim = CNN_data.shape[1:]
flatten_length = int(np.prod(CNN_data.shape[1:]))

#convert twist to numpy array, take only yaw
twist = np.asarray(twist)#[:, 1][:,None]

#define functions for initializing slightly positive random variables (TF_VARIABLES)
def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.02)
  return tf.Variable(initial, name = name)

def bias_variable(shape, name):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial, name = name)

#CREATE GRAPH #############

#define our placeholder variables for defining the symbolic expression to diff
#input is a (seq_size, 8, 8, 14) vector or something like that
x = tf.placeholder(tf.float32, shape=(None,) + image_dim)
x_flat = tf.reshape(x, [-1]) #smash everything into a flat vector (siamese NN)

y_data = tf.placeholder(tf.float32, shape=[None, output_num])

#run it through several FC dense layers
fc6_hidden_size = 500
global fc6W
fc6W = weight_variable([flatten_length, fc6_hidden_size], 'fc6W')
global fc6b
fc6b = bias_variable([fc6_hidden_size], 'fc6b')

fc6 = tf.nn.relu(tf.matmul(x_flat, fc6W) + fc6b)#I think we should do this like in alexnet

#apply dropout
keep_prob = tf.placeholder(tf.float32)
fc6_drop = tf.nn.dropout(fc6, keep_prob)

#final readout layer
global y_W
y_W = weight_variable([fc6_hidden_size, output_num], 'yW')
global y_B
y_B = bias_variable([output_num], 'yB')

#no softmax needed, included in the loss function part
y_pred = tf.matmul(fc6_drop, y_W) + y_B

#define loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_pred, y_data))

#define training step
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#start the Tensorflow session and initialize variables
global sess
sess = tf.Session()
sess.run(tf.initialize_all_variables())

global error_rates
error_rates = [0.]*num_epochs

for e in range(0, num_epochs):
    #create random permutation of indices in the data, twist arrays
    p = np.random.permutation(num_batches)

    # if no loss val is ever updated, show something is wrong
    loss_val = -999999

    #for each random index
    for index in p:

        #make sure we dont go below 0 in indexing
        if index - seq_size >= 0:
            #each batch is the random index and all seq_size images before that
            x_batch = CNN_data[(index - seq_size):index]
            y_batch = twist[(index - seq_size):index]

            #train
            _, loss_val = sess.run([train_step, cross_entropy], feed_dict={x: x_batch, y_data: y_batch, keep_prob: keep_probability})


    print "epoch: ", e, "loss: ", loss_val
    error_rates[e] = loss_val

#final training accuracy
pred, corr = sess.run([y_pred, y_data], feed_dict = {x:CNN_data, y_data:twist, keep_prob: 1})
print "final predictions:", pred, "correct:", corr

# SAVE DATA

#saver = tf.train.Saver([fc6W, fc6b, fc7W, fc7b, y_W, y_B])
#saver.save(sess, 'dnn_model')




