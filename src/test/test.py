import numpy as np

qq = np.transpose(np.array([
                    [0, 1, 2],
                    [3, 4, 5],
                    [6, 7, 8],
                     ]))

print(qq)
print(type(qq))

imageSize = np.array([480	,640])[::-1]
print(imageSize)