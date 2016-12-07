#!/usr/bin/env python

import rospy

from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Image
import numpy as np
from numpy import *
import time

from predictnet import Predictor

net_data = load("bvlc_alexnet.npy").item()
dnn_net_data = load("dnn_net_data.npy").item()

global predictionNet
predictionNet = Predictor(net_data, dnn_net_data)
global predictor
predictor = predictionNet.get_session()
   
Im1 = np.zeros((250,250,3));
Im2 = np.zeros((250,250,3));
Im3 = np.zeros((250,250,3));
Im4 = np.zeros((250,250,3));
ImCurr = np.zeros((250,250,3));

#timer stuff
t_last = time.time();
recordRate = 3; #hz
t_step = float(1)/recordRate;

#callback for images
def image_cb(data):
	global t_last
        t_dur = time.time() - t_last;
        t_valid = t_dur > t_step;
        #print t_dur, t_step, t_valid
        if t_valid:
            picture = data.data;
	    picture = np.frombuffer(picture, np.uint8);
	    picture = np.reshape(picture,(250,250,3));

	    global Im1
	    global Im2
	    global Im3
	    global Im4
	    global ImCurr

	    Im4 = Im3;
	    Im3 = Im2;
	    Im2 = Im1;
	    Im1 = ImCurr;
	    ImCurr = picture;

	    print "4:", Im4[1,1,1], "3:", Im3[1,1,1], "2:", Im2[1,1,1], "1:", Im1[1,1,1]

	    pos_twists = [[.125, -.2],[.125, .2],[.5, -.1],[.5, -.1],[.5, .05],[.5, .1],[1., -.05],[1., 0.],[1., .05]]

	    #=====calling the neural network=====

	    feed_list = [Im4, Im3, Im2, Im1, ImCurr]
	    output = predictor.run(predictionNet.prediction, feed_dict={predictionNet.x: feed_list}) 
	    index = np.argmax(output[-1])
	    dX = pos_twists[index][0]
	    dYaw = pos_twists[index][1]
	    #======================================

	    xVel = dX/2.
	    yVel = 0;
	    zVel = 0;

	    R_dot = 0;
	    P_dot = 0;
	    Y_dot = dYaw

	    lin = Vector3(xVel,yVel,zVel);
	    ang = Vector3(R_dot,P_dot,Y_dot);

	  
	    if np.any(Im4):
		twistOut = Twist(lin,ang);
	    else:
		twistOut = Twist(Vector3(0,0,0),Vector3(0,0,0));

	    TwPub.publish(twistOut);
	    t_last = time.time();

def main():
    rospy.init_node('neuro_drive');
    
    global TwPub
    #TwPub = rospy.Publisher('/husky_velocity_controller/cmd_vel', Twist);
    TwPub = rospy.Publisher('/NN_out/cmd_vel', Twist);
    
    ImSub = rospy.Subscriber("/camera/image_raw",Image, image_cb, queue_size = 1);

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin();

if __name__ == '__main__':
    main();
