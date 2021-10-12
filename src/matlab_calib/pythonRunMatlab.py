
###########################################################
# cd /usr/local/MATLAB/R2021a/extern/engines/python
# sudo python setup.py install
###########################################################

import os
print('Current Directory:', os.path.abspath(os.getcwd()))
print('file Directory:', os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
print('Change Current Directory To:', os.path.abspath(os.getcwd()))

import matlab.engine
eng = matlab.engine.start_matlab()
eng.test(nargout=0) # test.m
eng.quit()