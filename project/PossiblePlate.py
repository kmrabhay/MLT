
import cv2
import numpy as np

class PossiblePlate:

    def __init__(self):
	self.thresh = None

        self.rrLocationOfPlateInScene = None

        self.strChars = ""
    
        self.plate = None
        self.gray = None
      



