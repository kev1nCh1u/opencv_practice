import numpy as np
import glob
import math
from matplotlib import pyplot as plt


qq = np.transpose(np.array([
                    [[0, 1, 2]],
                    [[3, 4, 5]],
                    [[6, 7, 8]],
                     ]))
print(qq)

print(np.reshape(qq, (3,-1)))

print(np.reshape(qq, (3,-1))[:,1])

x = [1, 10]
y = [1, 10]
plt.plot(x,y, 'o', markersize=2)
plt.xlim([0, 640])
plt.ylim([480, 0])
plt.show()
