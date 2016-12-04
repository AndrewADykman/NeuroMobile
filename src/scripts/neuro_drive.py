#!/usr/bin/env python

import rospy

from geometry_msgs.msg import Twist
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Image
import numpy as np

#=========================================================
#import NEURAL_NETWORK as ASnjioknkmkkkkkkkkkkkkDF_NN
#=========================================================


#callback for images
def image_cb(data):

    picture = data.data;
    picture = np.frombuffer(picture, np.uint8);
    picture = np.reshape(picture,(500,500,3));

    #=====calling the neural network!!=====
    [X,Yaw] = NN.predict(picture);
    #======================================

    xVel = X;
    yVel = 0;
    zVel = 0;

    R_dot = 0;
    P_dot = 0;
    Y_dot = Yaw;

    lin = Vector3(xVel,yVel,zVel);
    ang = Vector3(R_dot,P_dot,Y_dot);

    twistOut = Twist(lin,ang);
    TwPub.publish(twistOut);

def main():
    rospy.init_node('neuro_drive');
    
    global TwPub
    TwPub = rospy.Publisher('/NN_out/cmd_vel', Twist);
    ImSub = rospy.Subscriber("/camera/image_raw",Image, image_cb, queue_size = 1);


    # ====== loading and initializing the neural network ======
    # global NN;
    # NN = new NN;
    # NN.loadWeights or initialize or whatever; 
    # =========================================================



    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin();

if __name__ == '__main__':
    main();
