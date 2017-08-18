'''
@brief: analysis the global shark attacks data from
https://www.kaggle.com/teajay/global-shark-attacks/downloads/attacks.csv
@author: Shylock Hg
@time: 2017/8/18
@email:tcath2s@icloud.com
'''

import pandas as pd
import matplotlib.pyplot as plt

f_data = 'attacks.csv'



#read data from attacks.csv
df = pd.read_csv(f_data, error_bad_lines=False)

#
acts = {}
for d in df['Activity']:
    if d in acts:
        acts[d] += 1
    else:
        acts[d] = 1

k = [k for k in acts]
v = [acts[key] for key in acts]

plt.figure(1)
plt.xlabel('Activity')
plt.ylabel('Num')
plt.bar(range(len(v)),v)
plt.show()
