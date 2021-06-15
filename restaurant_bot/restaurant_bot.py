# Import libraries
import webbrowser
from bs4 import BeautifulSoup
import urllib.request
 
def restaurant_bot(): 
  """
  1. Bot asks user for type of food and location
  2. Returns with recommendation in browser
  """

# User input for search query
  food_type = input("Hello! I am restaurant_bot and I can recommend you a highly rated place to eat!"
                    "\n\nWhat do you feel like eating? (Ex. Big Juicy Burger) ")
  location = input("\nWhere are you located? (Ex. San Francisco) ")
    
  food_type = food_type.replace(" ", "-")
  location = location.replace(" ", "-")
    
  results = ('https://www.yelp.com/search?find_desc=' + food_type + '&find_loc=' + location + '&sortby=review_count')
 
# Web scrape query page and select the URL for the top rated option
  parser = 'html.parser'
  resp = urllib.request.urlopen(results)
  soup = BeautifulSoup(resp, parser, from_encoding=resp.info().get_param('charset'))
  eatery_list = []

  for link in soup.select("a[href*='/biz']"):
    eatery_list.append(link['href'])
 
# Open option in browser for the user
  webbrowser.open('https://www.yelp.com' + eatery_list[0])

restaurant_bot()
