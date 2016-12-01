import numpy as np
import cv2
import pickle

with open('images.pickle','r') as f:
  images = pickle.load(f)

with open('twist.pickle','r') as g:
  twist = pickle.load(g)

shapely = np.reshape(images[8], [500, 500, 3])

cv2.imshow('c', shapely)
cv2.waitKey()
