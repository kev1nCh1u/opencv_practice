import numpy as np

qq = {'dis':1, 'log':2}

print(qq['dis'])

if __debug__:
    print ('Debug ON')
else:
    print ('Debug OFF')

disArr = np.arange(25).reshape(5, 5)
print()
print(disArr)
print()

j = 1

print()
print(disArr[4][j])
print()

disArr[4][j], disArr[4][j + 1] = disArr[4][j + 1], disArr[4][j]
print()
print(disArr)
print()