#!/usr/bin/env python

import rospy
import time
import pickle
import atexit
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
import numpy as np


#define input messages
twistIn = None;
imageIn = None;

#lists on data
instImage = [];
instTwist = [];

#timer stuff
t_last = time.time();
recordRate = 3; #hz
t_step = float(1)/recordRate;

#data dump on exit
@atexit.register
def pickleInstances():
    f = open("images.pickle", "w")
    pickle.dump(instImage, f);
    f.close();
    
    f = open("twist.pickle", "w")
    pickle.dump(instTwist, f);
    f.close();


#callback for twists
def twist_cb(data):

    twistIn = data;
    
    global X_lin_Z_ang;
    X_lin_Z_ang = [data.linear.x, data.angular.z];
    
    global twistNew;
    twistNew = True;
    
    t_dur = time.time() - t_last;
    t_valid = t_dur > t_step;
    
    if(twistNew and imageNew and t_valid):
        buildInstance();

#callback for images
def image_cb(data):

    global picture;
    picture = data.data;
    picture = np.frombuffer(picture, np.uint8);
    picture = np.reshape(picture,(500,500,3));
    global imageNew;
    imageNew = True;

    t_dur = time.time() - t_last;
    t_valid = t_dur > t_step;

    if(twistNew and imageNew and t_valid):
        buildInstance();

#combined result of both callbacks being called 
def buildInstance():

    instImage.append(picture);
    instTwist.append(X_lin_Z_ang);

    twistNew = False;
    imageNew = False;
    global t_last;
    t_last = time.time();

def main():
    rospy.init_node('instace_matcher');
    TwSub = rospy.Subscriber("/husky/cmd_vel", Twist, twist_cb);
    ImSub = rospy.Subscriber("/camera/image_raw",Image, image_cb, queue_size = 1);
    #ImSub = rospy.Subscriber("/camera/image_raw/compressed",CompressedImage, image_cb, queue_size = 1);

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin();

if __name__ == '__main__':
    main();
