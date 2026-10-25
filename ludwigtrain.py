import kagglehub
import pandas as pd
import os
import ray
from ludwig.api import LudwigModel

if not os.path.exists('path.txt'):
    kagglehub.login()
    path = kagglehub.competition_download('titanic')
    with open('path.txt', 'w') as f:
        f.write(path)

with open('path.txt', 'r') as f:
    path = f.read().strip()


df = pd.read_csv(os.path.join(path, 'train.csv'))

# Manually Configuring the model with ludwig
# 1. Define explicit Ludwig config
config = {
    "input_features": [
        {"name": "Pclass", "type": "category"},
        {"name": "Sex", "type": "category"},
        {"name": "Age", "type": "number", "preprocessing": {"missing_value_strategy": "fill_with_mean"}},
        {"name": "SibSp", "type": "number"},
        {"name": "Parch", "type": "number"},
        {"name": "Fare", "type": "number"},
        {"name": "Embarked", "type": "category"}
    ],
    "output_features": [
        {"name": "Survived", "type": "category"}
    ],
    "trainer": {
        "epochs": 10
    }
}

# 2. Initialize and train the model directly
model = LudwigModel(config=config, logging_level=20)
results = model.train(dataset=df, skip_save_model=True)

# 3. print the training results
print("Training results:", results)

# 4. save the model weights if needed
model.save('titanic_model')

print("Training completed successfully; model weights were saved.")