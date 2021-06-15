#load libraries
import pandas as pd 
from pandas import DataFrame
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

#read csv data
data = pd.read_csv(r'C:\Users\keith\OneDrive\Desktop\Python Scripts\Data Science\tripadvisor_hotel_reviews.csv', dtype={'Review':'str'})

df = DataFrame(data, )
df = df.sample(n=150)

#create new binary column of T/F if the user review contains the word excellent
df['Excellence'] = df['Review'].str.contains('excellent')
df['Horrendous'] = df['Review'].str.contains('horrible')

#find out what the most common words used are
common = Counter(" ".join(df['Review']).split()).most_common(50)

#change from T/F to 0/1 in order to create bar chart
#df['Excellence'] *= 1

print(common)
print(df.head(10))

plt.style.use('ggplot')
df['Excellence'].value_counts().plot(kind='bar', title='Does the Review Contain the Word Excellent?')
plt.xticks(rotation=45)

plt.show()

df['Horrendous'].value_counts().plot(kind='bar', title='Does the Review Contain the Word Horrible?')
plt.xticks(rotation=45)

plt.show()

bins = np.arange(1,6.1) - 0.5

df2 = DataFrame(data,columns=['Rating'])
df2.plot.hist(bins=bins, alpha=0.5, color='blue', edgecolor='black')
plt.title('Hotel Ratings')
plt.xticks(range(1,6,1))

plt.show()
