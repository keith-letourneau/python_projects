#!/usr/bin/env python
# coding: utf-8

class Roster:
    pcount = 0
    
    def __init__(self, name, position, number, goals):
        self.name = name
        self.position = position
        self.number = number
        self.goals = goals
        Roster.pcount += 1
        
        if Roster.pcount > 3:
            print('Max Number of Players Reached.', self.name, 'Cannot be added to the Roster.')
        
    def goals_scored(self):
        print(self.name, 'scored:', self.goals, 'goals')
    
    def as_dict(self):
        return {'Name': self.name, 'Position': self.position, 'Number': self.number, 'Goals': self.goals}

p1 = Roster('Sidney', 'Center', 87, 45)
p2 = Roster('Z', 'Defensemen', 33, 13)
p3 = Roster('Ovi', 'Left Wing', 8, 38)

p4 = Roster('Conor', 'Center', 97, 50)
p5 = Roster('Patrick', 'Right Wing', 29, 33)

print('Number of Players on Roster:', Roster.pcount)

p1.goals_scored()
p4.goals_scored()

import pandas as pd
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid")

player_list = [p1, p2, p3, p4, p5]

df = pd.DataFrame([player.as_dict() for player in player_list])
df = df.sort_values(by=['Goals'], ascending=False)
df.style.hide_index()

#change dpi to 300 for higher resolution plot
plt.figure(dpi=90)
sns.barplot(x=df['Name'], y=df['Goals'])
plt.xlabel('Player Name')
plt.ylabel('Goals Scored')
plt.title('Top NHL Goal Scorers 2021')
plt.axhline(df['Goals'].mean(), color='red', alpha=.5, ls='--')
plt.show()

print(Roster.__dict__)
