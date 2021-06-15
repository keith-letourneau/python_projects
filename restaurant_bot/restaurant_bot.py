#restaurant_bot

import webbrowser
from bs4 import BeautifulSoup
import urllib.request
 
def eatery(): 
  """
  Bot asks user for type of food and location
  returns with recommendation in browser
  """

#User input for search query
  food_type = input("Hello! I am the restaurant_bot and I can recommend you a highly rated place to eat!\n\nI have only one rule: no spaces are allowed in your answers!\n\nWhat do you feel like eating? (Ex. Korean-BBQ) ")

  location = input("\nWhere are you located? (Ex. San-Francisco) ")

  results = ('https://www.yelp.com/search?find_desc=' + food_type + '&find_loc=' + location + '&sortby=review_count')
 
#Web scrape query page and select the URL for the top rated option
  parser = 'html.parser'

  resp = urllib.request.urlopen(results)

  soup = BeautifulSoup(resp, parser, from_encoding=resp.info().get_param('charset'))

  eatery_list = []

  for link in soup.select("a[href*='/biz']"):
    eatery_list.append(link['href'])
 
#Open option in browser for the user
  webbrowser.open('https://www.yelp.com' + eatery_list[0])

eatery()