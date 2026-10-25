import kagglehub
import pandas as pd
import os
from ludwig.automl import auto_train

if not os.path.exists('path.txt'):
    kagglehub.login()
    path = kagglehub.competition_download('titanic')
    with open('path.txt', 'w') as f:
        f.write(path)

with open('path.txt', 'r') as f:
    path = f.read().strip()


df = pd.read_csv(os.path.join(path, 'train.csv'))

# ray_temp = os.path.join(os.environ["LOCALAPPDATA"], "ray_tmp")
# os.makedirs(ray_temp, exist_ok=True)

# if not ray.is_initialized():
#     ray.init(_temp_dir=ray_temp, include_dashboard=False)

auto_train_results = auto_train(
    dataset=df,
    target="Survived",
    time_limit_s=600,
    output_directory=r"C:\Users\chris\Documents\Mlops\KaizenML-Grup2\results\run_1",
    resume=False,
    skip_save_progress=True,
    user_config={
        "hyperopt": {
            "executor": {
                "num_samples": 2,
                "max_concurrent_trials": 2,
            }
        }
    },
)

print("AutoML training completed!")
print(auto_train_results)

