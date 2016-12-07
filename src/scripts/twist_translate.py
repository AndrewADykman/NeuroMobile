import pickle
import argparse
import numpy as np


possible_twists = [[.125, -.2],[.125, .2],[.5, -.1],[.5, -.1],[.5, .05],[.5, .1],[1., -.05],[1., 0.],[1., .05]]

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
  new_twists[j][best_twist] = 1
  j += 1

with open(args.out_filename, 'wb') as f:
  pickle.dump(new_twists, f)
