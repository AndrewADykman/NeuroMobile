import numpy as np

class NeuralLayer:
  def __init__(self, num_inputs, num_neurons):
    self.num_inputs = num_inputs
    self.num_neurons = num_neurons
    self.potentials = np.zeros([self.num_neurons, 1])
    self.transform_matrix = np.random.rand(self.num_inputs, self.num_neurons)

  def setPotentials(self, input_potentials):
    self.potentials = np.dot(input_potentials, self.transform_matrix)

  def getPotentials(self):
    return self.potentials

  def backPropogate(self, guilt_vector):
    #alter transform matrix
    #create pass-back guilt vector

  def setAsInputLayer(self):
      self.transform_matrix = np.identity(num_inputs)
