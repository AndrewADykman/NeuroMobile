import numpy as np
import tensorflow as tf

class RawTrData:
  def __init__(self, image, x, yaw):
    self.image = image
    self.x = x
    self.yaw = yaw
    
  def getAlexTrData(self):
    pp = #howthefuckdowedothis
    alexTrData = AlexTrData(pp, self.x, self.yaw)

class AlexTrData:
  def __init__(self, pp, x, yaw):
    self.pp = pp
    self.x = x
    self.yaw = yaw
