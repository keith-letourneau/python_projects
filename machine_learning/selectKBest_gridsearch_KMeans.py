import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score,mean_squared_error
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, silhouette_score
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering


df = pd.read_csv(r'C:\Users\keith\Downloads\manhattan.csv')
df = df.drop(['rental_id','borough'], axis=1)
df = pd.get_dummies(df)

x = df.drop('rent', axis=1)
x = sm.add_constant(x)
y = df['rent']

lm = sm.OLS(y, x).fit()

print(lm.summary())

x = df.drop(['rental_id', 'rent','bedrooms','bathrooms','size_sqft','floor','building_age_yrs','has_roofdeck','has_washer_dryer','has_doorman','has_elevator','has_dishwasher','has_patio','has_gym'], axis=1)
x = sm.add_constant(x)
y = df['rent']
lm = sm.OLS(y, x).fit()

print(lm.summary())

model = LinearRegression()

x = df.drop('rent', axis=1)
y = df['rent']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3, random_state = 101)

model.fit(x_train,y_train)

y_pred=model.predict(x_test)
mse=mean_squared_error(y_test, y_pred)
rmse=mse**0.5

print("Without SelectKBest \n")
print("R squared is",r2_score(y_test, y_pred))
print("rmse is", rmse)

#with SelectKBest features
x = df.drop('rent', axis=1)
y = df['rent']

model = LinearRegression()

#with SelectKBest features
for i in range(1,47):
    bestfeatures = SelectKBest(score_func=f_regression, k=i)
    new_x = bestfeatures.fit_transform(x,y)
    x_train, x_test, y_train, y_test = train_test_split(new_x,y,test_size=0.3, 
                                                        random_state=1)
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    mse=mean_squared_error(y_test, predictions)
    rmse=mse**0.5
    print(i, rmse)

bestfeatures = SelectKBest(score_func=f_regression, k=28)
new_x = bestfeatures.fit_transform(x,y)
x_train, x_test, y_train, y_test = train_test_split(new_x,y,test_size=0.3, 
                                                        random_state=1)
model.fit(x_train, y_train)
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test, y_pred)
rmse=mse**0.5

print("R squared is",r2_score(y_test, y_pred))
print("rmse is", rmse)

best = bestfeatures.get_support()
best = best.tolist()
col = df.columns[1:].tolist()

lm_columns = pd.DataFrame(best, columns = ['best_feature'])
lm_columns['variable'] = col
lm_columns = lm_columns.sort_values(['best_feature','variable'], ascending=[False,True])
lm_columns.index = np.arange(1, 47)
lm_columns[:28]

km2 = KMeans(n_clusters=3, random_state=0)
km2.fit(df)
km2.labels_

df['labels'] = km2.labels_

sns.scatterplot(x="rent",y="size_sqft", data=df, hue="labels", palette="Set1")

wcv = []
silk_score = []

for i in range(2,11):
    km = KMeans(n_clusters=i, random_state=0)
    km.fit(df)
    wcv.append(km.inertia_)
    silk_score.append(silhouette_score(df, km.labels_,))
    
plt.plot(range(2,11), wcv)
plt.xlabel('no. of clusters')
plt.ylabel('within cluster variation')
plt.show()

hc = AgglomerativeClustering(n_clusters=2, linkage='single')
hc.fit(df)

df['labels'] = hc.labels_

sns.scatterplot(x="rent",y="size_sqft", data=df, hue="labels", palette="Set1")

df.loc[df['rent'] > 15000, 'label'] = 0
df.loc[df['rent'] < 12000, 'label'] = 1
df.loc[df['rent'] < 8000, 'label'] = 2
df.loc[df['rent'] < 5000, 'label'] = 3
sns.scatterplot(x="rent",y="size_sqft", data=df, hue="label", palette="Set1")

sns.scatterplot(x="rent",y="building_age_yrs", data=df, hue="label", palette="Set1")

df = pd.read_csv(r'C:\Users\keith\Downloads\manhattan.csv')
df = df.drop(['rental_id','borough'], axis=1)
df = pd.get_dummies(df)

km = KMeans(n_clusters=3, random_state=1)
km.fit(df)

df['labels'] = km.labels_

sns.scatterplot(x="rent",y="size_sqft", data=df, hue="labels", palette="Set1")
sns.pairplot(df,hue='labels')

