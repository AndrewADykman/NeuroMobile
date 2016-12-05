import pickle
import numpy as np

with open('train_images.pickle', 'r') as f:
  train_images = pickle.load(f)

with open('train_twists.pickle', 'r') as g:
  train_twists = pickle.load(g)


for i in range(0, len(train_images)):
	if np.shape(train_images[i]) != (500, 500, 3):
		print "bad one"
		train_images[i] = np.reshape(train_images[i], (500, 500, 3))
		
			
np.stack(train_images, axis = 0)

pickle.dump(train_images, open('train_images.pickle', 'wb'))

'''
with open('test_images.pickle', 'r') as h:
  test_images = pickle.load(h)

with open('test_twists.pickle', 'r') as i:
  test_images = pickle.load(i)
'''
