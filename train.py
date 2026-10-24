import kagglehub
import pandas as pd
import os
import ray
from ludwig.api import LudwigModel
from ludwig.automl import auto_train

if not os.path.exists('path.txt'):
    kagglehub.login()
    path = kagglehub.competition_download('titanic')
    with open('path.txt', 'w') as f:
        f.write(path)

with open('path.txt', 'r') as f:
    path = f.read().strip()

# if not ray.is_initialized():
#     ray.init(num_cpus=1, ignore_reinit_error=True, include_dashboard=False)

# os.environ["RAY_num_cpus"] = "1"

df = pd.read_csv(os.path.join(path, 'train.csv'))

## There are two ways to train a model with ludwig or AutoML: manually configuring the model
#Automated training with AutoML
auto_train_results = auto_train(
    dataset=df,
    target='Survived',      
    time_limit_s=600,        
    tune_for_memory=False
)

print("AutoML training completed!")
print(auto_train_results)
# ray.shutdown()

