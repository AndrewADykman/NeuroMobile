import pickle
import argparse
import numpy as np


possible_twists = [[.125, -.1],[.125, .1],[.5, -.06],[.5, -.03],[.5, .03],[.5, .06],[1., -.03],[1., 0.],[1., .03]]

parser = argparse.ArgumentParser()
parser.add_argument('--in-filename', type=str, help="the input filename")
parser.add_argument('--out-filename', type=str, help="the output filename")
args = parser.parse_args()

with open(args.in_filename, 'r') as f:
  twists = pickle.load(f)

new_twists = np.zeros([len(twists), 9])

j = 0
for twist in twists:
  best_twist = -1
  best_dist = 99999
  for i in range(0, 9):
    dist = (twist[0] - possible_twists[i][0])**2 + (twist[1] - possible_twists[i][1])**2
    if dist < best_dist:
      best_dist = dist
      best_twist = i
  print 'iteration: ' + str(j) + 'b_t: ' + str(best_twist)
  new_twists[j][best_twist] = 1
  print new_twists[j]
  j += 1

print new_twists
with open(args.out_filename, 'wb') as f:
  pickle.dump(new_twists, f)
