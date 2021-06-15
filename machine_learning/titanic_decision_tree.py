#Kaggle Titianic ML Data Science Competition
#Decision Tree Algorithm Solution
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#load and visualize training dataset
df_train = pd.read_csv(r'C:\Users\keith\OneDrive\Desktop\Python Scripts\Data Science\train.csv')
df_train['Sex'] = pd.factorize(df_train.Sex)[0]
df_train = df_train[['Sex', 'Pclass', 'Age', 'Survived']]
df_train.dropna(inplace=True)

df_train.head(20)
df_train['Survived'].value_counts().plot(kind='bar', color='#9eadd6')

#fit decision tree model
x = df_train[['Sex', 'Pclass', 'Age']]
y = df_train['Survived']

dt_clf = tree.DecisionTreeClassifier(max_depth=5)
dt_clf.fit(x, y)

#load and clean test data for predicting outputs
df_test = pd.read_csv(r'C:\Users\keith\OneDrive\Desktop\Python Scripts\Data Science\test.csv')
df_test = df_test[['Sex', 'Pclass', 'Age']]
df_test['Sex'] = pd.factorize(df_test.Sex)[0]
df_test['Age'] = df_test['Age'].fillna((df_test['Age'].mean()))
df_test = df_test[['Sex', 'Pclass', 'Age']]

#use model to make predictions and save to new CSV file
prediction = dt_clf.predict(df_test)
df_test['Survived'] = prediction
df_test['PassengerId'] = df_test.index + 892
df_test = df_test[['PassengerId', 'Survived']]

df_test.head(10)
df_test['Survived'].value_counts().plot(kind='bar', color='#9eadd6')
#df_test.to_csv(r'C:\Users\keith\OneDrive\Desktop\submission.csv', index=False)

#plot
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)

tree.plot_tree(dt_clf, fig.savefig('imagename.png'))