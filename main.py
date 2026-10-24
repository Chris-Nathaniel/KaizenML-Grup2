import logging
import pprint
import kagglehub
import pandas as pd
import os
from ludwig.automl import auto_train
# #KGAT_cf52def530194463cec9a03109bd40d6
# kagglehub.login()
# check if the path.txt file exists else download the competition files and save the path to a text file

if not os.path.exists('path.txt'):
    path = kagglehub.competition_download('titanic')

    #save the path to a text file 
    with open('path.txt', 'w') as f:
        f.write(path)

# read the path from the text file
with open('path.txt', 'r') as f:
    path = f.read().strip()

df = pd.read_csv(os.path.join(path, 'train.csv'))
print(df.head())

auto_train_results = auto_train(
    dataset=df,
    target='survived',
    time_limit_s=7200,
    tune_for_memory=False
)

pprint.pprint(auto_train_results)





