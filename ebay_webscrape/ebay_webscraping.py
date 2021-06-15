#!/usr/bin/env python
# coding: utf-8

# In[61]:


import requests
from bs4 import BeautifulSoup
import urllib.request
import pandas as pd
from pandas import DataFrame

ebay_url = 'https://www.ebay.com/sch/i.html?_from=R40&_trksid=p2380057.m570.l1313&_nkw=secret+wars+8+cgc&_sacat=0'
reqs = requests.get(ebay_url)
soup = BeautifulSoup(reqs.text, 'lxml')

items = []

for item in soup.find_all('h3', attrs={'class':"s-item__title"}):
    items.append(item.text.strip())
    
prices = []

for price in soup.find_all('span', attrs={'class':"s-item__price"}):
    prices.append(price.text.strip())
    
shipping = []

for ship in soup.find_all('span', attrs={'class':"s-item__logisticsCost"}):
    shipping.append(ship.text.strip())
    
links = []

for link in soup.find_all('a', attrs={'class':"s-item__link"}):
    links.append(link['href'])
    
df = DataFrame(items,columns=['Item Name'])
df['Price'] = prices
df['Shipping'] = shipping
df['Product Link'] = links

df.to_excel(r'C:\Users\keith\OneDrive\Desktop\Python Scripts\comic-prices.xlsx', index=False)
df

