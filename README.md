import lightgbm as lgb
import pandas as pd
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load your dataset (update the path as needed)
data = pd.read_csv('/content/parkinsons.csv')
feature_cols = data.columns.drop('status').tolist()
X1 = data[feature_cols]  # Features
y1 = data.status  # Target variable

# Clean column names
X1 = X1.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
feature_cols = list(X1.columns)
