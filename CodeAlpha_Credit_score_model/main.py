from networkx import display
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# LOAD DATASET
df = pd.read_csv('data/heloc_dataset_v1.csv')
print(df.head())

# DATA CLEANING
# df.replace(' ', np.nan, inplace=True)