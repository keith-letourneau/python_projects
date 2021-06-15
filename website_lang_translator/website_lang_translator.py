#!/usr/bin/env python
# coding: utf-8


# # <center>Translate French News Website to English</center>
# 
# This script with translate h1,h2,h3 tags to English, place into list and save as .txt file.


#translate French news website headlines to English
from google_trans_new import google_translator  
import requests
from bs4 import BeautifulSoup
import urllib.request
import pandas as pd
from pandas import DataFrame

url = 'https://www.monde-diplomatique.fr/'
reqs = requests.get(url)
soup = BeautifulSoup(reqs.text, 'lxml')

news = []

for heading in soup.find_all(["h1", "h2", "h3"]):
    news.append(heading.text.strip())

translator = google_translator()

df = DataFrame(news,columns=['French'])
df['English'] = df['French'].apply(translator.translate, lang_src='fr', 
                                                         lang_tgt='en')
df.head(5)


#turn translations into a list and save to .txt file
fr_headlines = df['English'].values.tolist()
    
with open(r'C:\Users\keith\OneDrive\Desktop\Python Scripts\French Headlines.txt', 'w', 
          encoding="utf-8") as my_file:
    my_file.write('French Headlines from Today' + '\n' + '\n')
    for headline in fr_headlines:
        my_file.write(headline + '\n')