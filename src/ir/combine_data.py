import numpy as np
import glob
import math
from matplotlib import pyplot as plt
import pandas as pd

saveDataPath = 'data/point_path/data_concat.csv'

dfAll = pd.DataFrame()
for file in sorted(glob.glob('data/point_path/point*')):
    

    print(file)
    header_list = ["x"+file[-6:-4], "y"+file[-6:-4]]
    df = pd.read_csv(file, header=0, names=header_list, usecols=[1,2], skiprows=0)
    # print(df)
    # exit()
    
    dfAll = pd.concat([dfAll, df], axis=1)

print(dfAll)
pd.DataFrame(dfAll).to_csv(saveDataPath)
print('Save data file to:', saveDataPath)