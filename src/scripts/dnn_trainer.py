import tensorflow as tf
import numpy as np
import pickle
import argparse
import atexit

# DEFINE HYPERPARAMETERS
parser = argparse.ArgumentParser(description='take training arguments.')
parser.add_argument('--num-epochs', type=int, default=300, help='The number of epochs')
parser.add_argument('--keep-prob', type=float, default=.75, help='Probability of an edge being kept in the network each training iteration')
parser.add_argument('--mbatch-size', type=int, default=30, help='# of images per minibatch')
args = parser.parse_args()

@atexit.register
def saveData():

	print 'saving data...'

	np.save('error_rates.npy', error_rates)

	dnn_net_data = {}
	dnn_net_data['fc6'] = []
	dnn_net_data['fc6'].append(fc6W.eval(sess))
	dnn_net_data['fc6'].append(fc6b.eval(sess))

	dnn_net_data['fc7'] = []
	dnn_net_data['fc7'].append(fc7W.eval(sess))
	dnn_net_data['fc7'].append(fc7b.eval(sess))

	dnn_net_data['fc8'] = []
        dnn_net_data['fc8'].append(fc8W.eval(sess))
	dnn_net_data['fc8'].append(fc8b.eval(sess))

	dnn_net_data['y'] = []
	dnn_net_data['y'].append(y_W.eval(sess))
	dnn_net_data['y'].append(y_B.eval(sess))

	np.save('dnn_net_data.npy', dnn_net_data)

#keep batch size small
miniBatchSize = args.mbatch_size

num_epochs = args.num_epochs

keep_probability = args.keep_prob

output_num = 2

# OPEN PICKLED OUTPUT FROM ALEXNET
with open('CNN_filters_train_mamba.pickle','rb') as f:
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
twist = twist * 100 #change range from -0.5, 0.5 to -50, 50
#twist = twist.tolist()

#define functions for initializing slightly positive random variables (TF_VARIABLES)
def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.02)
  return tf.Variable(initial, name = name)

def bias_variable(shape, name):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial, name = name)

#CREATE GRAPH #############

#define our placeholder variables for defining the symbolic expression to diff
x = tf.placeholder(tf.float32, shape=(None,) + image_dim)
x_flat = tf.reshape(x, [-1, int(np.prod(x.get_shape()[1:]))])

y_data = tf.placeholder(tf.float32, shape=[None, output_num])

#run it through several FC dense layers
fc6_hidden_size = 12000
global fc6W 
fc6W = weight_variable([flatten_length, fc6_hidden_size], 'fc6W')
global fc6b 
fc6b = bias_variable([fc6_hidden_size], 'fc6b')

fc6 = tf.nn.relu(tf.matmul(x_flat, fc6W) + fc6b)#I think we should do this like in alexnet

#run it through several FC dense layers
fc7_hidden_size = 7000
global fc7W 
fc7W = weight_variable([fc6_hidden_size, fc7_hidden_size], 'fc7W')
global fc7b 
fc7b = bias_variable([fc7_hidden_size], 'fc7b')

fc7 = tf.nn.relu(tf.matmul(fc6, fc7W) + fc7b)#I think we should do this like in alexnet

#run it through several FC dense layers
fc8_hidden_size = 4000
global fc8W 
fc8W = weight_variable([fc7_hidden_size, fc8_hidden_size], 'fc8W')
global fc8b 
fc8b = bias_variable([fc8_hidden_size], 'fc8b')

fc8 = tf.nn.relu(tf.matmul(fc7, fc8W) + fc8b)#I think we should do this like in alexnet

#apply dropout
keep_prob = tf.placeholder(tf.float32)
fc8_drop = tf.nn.dropout(fc8, keep_prob)

#final readout layer (2 output nodes, dYaw and dx)
global y_W 
y_W = weight_variable([fc8_hidden_size, output_num], 'yW')
global y_B 
y_B = bias_variable([output_num], 'yB')

y_pred = tf.matmul(fc8_drop, y_W) + y_B

#define loss function
squared_loss = tf.reduce_mean(tf.square(y_pred - y_data))

#define training step
train_step = tf.train.AdamOptimizer(1e-4).minimize(squared_loss)

#start the Tensorflow session and initialize variables
global sess 
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
global error_rates 
error_rates = [0.]*num_epochs

for e in range(0, num_epochs):

    p = np.random.permutation(num_batches)
    shuffled_x = CNN_data[p]
    shuffled_y = twist[p]

    count = 0
    while (count + miniBatchSize < num_batches):
        x_batch = shuffled_x[count:(count+miniBatchSize)]
        y_batch = shuffled_y[count:(count+miniBatchSize)]
        count += miniBatchSize

        #train
        _, loss_val = sess.run([train_step, squared_loss], feed_dict={x: x_batch, y_data: y_batch, keep_prob: keep_probability})

    print "epoch: ", e, "loss: ", loss_val
    error_rates[e] = loss_val

#final training accuracy
pred, corr = sess.run([y_pred, y_data], feed_dict = {x:CNN_data, y_data:twist, keep_prob: 1})
print "final predictions:", pred, "correct:", corr

# SAVE DATA

#saver = tf.train.Saver([fc6W, fc6b, fc7W, fc7b, y_W, y_B])
#saver.save(sess, 'dnn_model')




