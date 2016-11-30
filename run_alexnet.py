
from altered_alexnet import *
import pickle

#load weights
weights = load("bvlc_alexnet.npy").item()
my_alex = AlexCNN(weights)

#input is one long concatenated list of images and a parallel list of twists
#TODO get data from ROS
image_list = #TODO
twist_list = #TODO

processed_list = my_alex.runDict(image_list)
#figure out how to save processed_list. can't pickle, have to use TF's saver function.






