import numpy as np
import glob

qq = np.transpose(np.array([
                    [0, 1, 2],
                    [3, 4, 5],
                    [6, 7, 8],
                     ]))

print(qq)
print(type(qq))

imageSize = np.array([480	,640])[::-1]
print(imageSize)


path = "img/stereo_calibration/new/"
imagesLeft = glob.glob(path + '1/*.jpg')
print(imagesLeft)