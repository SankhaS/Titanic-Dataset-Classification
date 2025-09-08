import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split

df= pd.read_csv("../data/titanic.csv")

x= df.drop(columns=['Survived','Name', 'Ticket', 'Cabin', 'PassengerId'])
y= df['Survived']

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2)

print("Training data shape: " , x_train.shape)
print("Testing data shape: " , x_test.shape)

