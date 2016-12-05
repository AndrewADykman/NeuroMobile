from numpy import *
import tensorflow as tf
import pickle

"""Predictor class -- implements the entire graph structure of alexnet + ournet.py for the purpose
of calculating predictions, given the loaded alexnet and ournet weights.

TODO -- make this a lot better"""

class Predictor:

    def __init__(self, net_data, dnn_net_data):
        image_dim = (500, 500, 3)
        self.x = tf.placeholder(tf.float32, (None,) + image_dim)

        # conv1
        # conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
        k_h = 11;
        k_w = 11;
        c_o = 96;
        s_h = 4;
        s_w = 4;
        conv1W = tf.Variable(net_data["conv1"][0])
        conv1b = tf.Variable(net_data["conv1"][1])
        conv1_in = self.conv(self.x, conv1W, conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=1)
        conv1 = tf.nn.relu(conv1_in)

        # lrn1
        # lrn(2, 2e-05, 0.75, name='norm1')
        radius = 2;
        alpha = 2e-05;
        beta = 0.75;
        bias = 1.0
        lrn1 = tf.nn.local_response_normalization(conv1,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)
        # maxpool1
        # max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
        k_h = 3;
        k_w = 3;
        s_h = 2;
        s_w = 2;
        padding = 'VALID'
        maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

        # conv2
        # conv(5, 5, 256, 1, 1, group=2, name='conv2')
        k_h = 5;
        k_w = 5;
        c_o = 256;
        s_h = 1;
        s_w = 1;
        group = 2
        conv2W = tf.Variable(net_data["conv2"][0])
        conv2b = tf.Variable(net_data["conv2"][1])
        conv2_in = self.conv(maxpool1, conv2W, conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv2 = tf.nn.relu(conv2_in)

        # lrn2
        # lrn(2, 2e-05, 0.75, name='norm2')
        radius = 2;
        alpha = 2e-05;
        beta = 0.75;
        bias = 1.0
        lrn2 = tf.nn.local_response_normalization(conv2,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias)
        # maxpool2
        # max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
        k_h = 3;
        k_w = 3;
        s_h = 2;
        s_w = 2;
        padding = 'VALID'
        maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

        # conv3
        # conv(3, 3, 384, 1, 1, name='conv3')
        k_h = 3;
        k_w = 3;
        c_o = 384;
        s_h = 1;
        s_w = 1;
        group = 1
        conv3W = tf.Variable(net_data["conv3"][0])
        conv3b = tf.Variable(net_data["conv3"][1])
        conv3_in = self.conv(maxpool2, conv3W, conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv3 = tf.nn.relu(conv3_in)

        # conv4
        # conv(3, 3, 384, 1, 1, group=2, name='conv4')
        k_h = 3;
        k_w = 3;
        c_o = 384;
        s_h = 1;
        s_w = 1;
        group = 2
        conv4W = tf.Variable(net_data["conv4"][0])
        conv4b = tf.Variable(net_data["conv4"][1])
        conv4_in = self.conv(conv3, conv4W, conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv4 = tf.nn.relu(conv4_in)

        # conv5
        # conv(3, 3, 256, 1, 1, group=2, name='conv5')
        k_h = 3;
        k_w = 3;
        c_o = 256;
        s_h = 1;
        s_w = 1;
        group = 2
        conv5W = tf.Variable(net_data["conv5"][0])
        conv5b = tf.Variable(net_data["conv5"][1])
        conv5_in = self.conv(conv4, conv5W, conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
        conv5 = tf.nn.relu(conv5_in)

        # maxpool5
        # max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
        k_h = 3;
        k_w = 3;
        s_h = 2;
        s_w = 2;
        padding = 'VALID'
        maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

        ## OURNET PART OF THE GRAPH ##############
        # define our placeholder variables for defining the symbolic expression to diff
        maxpool_flat = tf.reshape(maxpool5, [-1, int(prod(maxpool5.get_shape()[1:]))])

        # run it through several FC dense layers
        fc6W = tf.Variable(dnn_net_data['fc6'][0])
        fc6b = tf.Variable(dnn_net_data['fc6'][1])

        fc6 = tf.nn.relu(tf.matmul(maxpool_flat, fc6W) + fc6b)

        fc7W = tf.Variable(dnn_net_data['fc7'][0])
        fc7b = tf.Variable(dnn_net_data['fc7'][1])

        fc7 = tf.nn.relu(tf.matmul(fc6, fc7W) + fc7b)

        # final readout layer (2 output nodes, dYaw and dx)
        y_W = tf.Variable(dnn_net_data['y'][0])
        y_B = tf.Variable(dnn_net_data['y'][1])

        self.prediction = tf.matmul(fc7, y_W) + y_B

        self.sess = tf.Session()
        self.sess.run(tf.initialize_all_variables())

    # CONV FUNCTION
    def conv(self, input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
        '''From https://github.com/ethereon/caffe-tensorflow
        '''
        c_i = input.get_shape()[-1]
        assert c_i % group == 0
        assert c_o % group == 0
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

        if group == 1:
            conv = convolve(input, kernel)
        else:
            input_groups = tf.split(3, group, input)
            kernel_groups = tf.split(3, group, kernel)
            output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
            conv = tf.concat(3, output_groups)
        return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])

    def get_session(self):
        return self.sess

    def close_session(self):
        return self.sess.close()


#MAIN SESSION------------------------------------
'''
#LOAD ALEXNET WEIGHTS
net_data = load("bvlc_alexnet.npy").item()

#LOAD OUR WEIGHTS
dnn_net_data = load("dnn_net_data.npy").item()

#import images from ROS
with open('images.pickle','r') as f:
  images2 = pickle.load(f)

with open('twist.pickle','r') as g:
  twist2 = pickle.load(g)

images2 = images2
twist2 = asarray(twist2) * 100

#create our thing
my_pred = Predictor(net_data, dnn_net_data)

print shape(images2)

for i in range(0, len(images2)):
    image = list()
    image.append(images2[i])
    prediction = my_pred.get_session().run(my_pred.prediction, feed_dict={my_pred.x: image})
    print "correct:", twist2[i]
    print "guessed:", prediction

my_pred.close_session()


'''
