import numpy as np
import glob
import math
from matplotlib import pyplot as plt
import cv2

fs = cv2.FileStorage("param/matlab_stereo_param.yaml", cv2.FILE_STORAGE_READ)

IntrinsicMatrix1 = fs.getNode("IntrinsicMatrix1").mat()
RadialDistortion1 = fs.getNode("RadialDistortion1").mat()
TangentialDistortion1 = fs.getNode("TangentialDistortion1").mat()

IntrinsicMatrix1 = np.transpose(IntrinsicMatrix1)
print(IntrinsicMatrix1)


distCoeffs1 = np.concatenate((RadialDistortion1, TangentialDistortion1), axis=1)
print(distCoeffs1)

