import kagglehub
import pandas as pd
import os
from ludwig.api import LudwigModel

if not os.path.exists('path.txt'):
    kagglehub.login()
    path = kagglehub.competition_download('titanic')
    with open('path.txt', 'w') as f:
        f.write(path)

with open('path.txt', 'r') as f:
    path = f.read().strip()

df = pd.read_csv(os.path.join(path, 'train.csv'))

# 1. Clean data and handle missing values
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
df = df.dropna(subset=['Survived'])
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Ensure target is string/category for classification
df['Survived'] = df['Survived'].astype(str)

# 2. Define explicit Ludwig config
config = {
    "input_features": [
        {"name": "Pclass", "type": "category"},
        {"name": "Sex", "type": "category"},
        {"name": "Age", "type": "number"},
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

# 3. Initialize and train the model directly
model = LudwigModel(config=config, logging_level=20)
results = model.train(dataset=df, skip_save_model=True)

print("Training completed successfully; model weights were not saved.")