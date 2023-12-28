import numpy as np
import os 
import pandas as pd



A = np.array([[2,3],[1,4]])

df = pd.DataFrame(A)
path = os.path.abspath('Results/x.csv')
df.to_csv(path)

B = np.array(pd.read_csv(path))
C = B[:,1:]

i=1

file_name = 'Res/test_number_{}/test.txt'.format(i) 
os.makedirs(os.path.dirname(file_name), exist_ok=True)

with open(file_name,'w+') as f:
    f.write('Fuck this \n')
    f.write('Fuck that')
    f.close()

