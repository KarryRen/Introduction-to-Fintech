import pandas as pd
import numpy as np
import os

path_040_lst = []
for f in os.listdir('TRD_Dalyr'):
#    for f in os.listdir('TRD_Dalyr/' + files):
       if f.split('.')[-1] == 'xlsx':
           path_040_lst.append('/'.join(['TRD_Dalyr',f]))
print(path_040_lst)