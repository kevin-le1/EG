import pandas as pd
import pyarrow as pyarrow
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Polygon, Point
import json
import numpy as np
# the unused imports are commented out because they have been used once to allocate data to text files

print("======================================================================================================================")
print("1. Write a python class called ProcessGameState that will accomplish the following:")
print("a. Handle file ingestion and ETL (if deemed necessary)")
# Below is how I am handling file ingestion and how I am extracting and transforming the data (specifically with use of pandas).
# allows for all columns to be seen in terminal
pd.options.display.max_columns = None
pd.options.display.max_rows = None
info = pd.read_parquet('game_state_frame_data.parquet', engine='pyarrow')


# code to observe the data
print(info.head(10))
print(info.tail(10))
# checking the shape of the data
print(info.shape)
# keeping track of total number of rows
shapeSize = 221330
print(info.columns)

print("======================================================================================================================")

print("b. Return whether or not each row falls within a provided boundary")
# allocating columns/rows so value manipulation is easier
boundaryX = info[["x"]]
boundaryY = info[["y"]]
boundaryZ = info[["z"]]

# utilizing gpd (geoseries) to create a polygon to act as the boundary
box = gpd.GeoSeries(
    Polygon([(-1735, 250), (-2024, 398), (-2806, 742), (-2472, 1233), (-1565, 580)])
)

# This is the code that has been excecuted to generate the text file for boolean results
"""
file = open('test3.txt', 'w')
i = 0
while i < 221330:
    file.write(str(box.contains(Point(boundaryX.loc[i].astype('int'), boundaryY.loc[i].astype('int')) & (285 <= boundaryZ.loc[i].astype('int') <= 421).bool())))
    i+=1
file.close()
"""

# unit testing
t = 22926
print(boundaryX.loc[t].astype('int'))
print(boundaryY.loc[t].astype('int'))
print("Should be false in booleanboundary.txt because the bounds of the y value is a lot lower than the y values of the boundary")

print("The boolean expression of whether each row falls within a provided boundary is in the booleanboundary.txt file.")

print("======================================================================================================================")

print("c. Extract the weapon classes from the inventory json column")

# allocating data and observing a row of the json/viewing the shape of the data
inventory = info[["inventory"]]
print(json.dumps(str(inventory.head(1))) + " data within inventory")
print(inventory.shape)

# Below is the code to extract the weapon classes from inventory column
"""
file = open('inventory.txt','w')
for i in list(inventory):
    file.write(f"{inventory[i]}+ \n")
file.close()
"""

print("The extraction of the inventory is in the inventory.txt file.")


print("======================================================================================================================")
print("2. Using the created class, answer the following questions:")


print("a. Is entering via the light blue boundary a common strategy used by Team2 on T (terrorist) side?")

j = 0
cat_cols = info.select_dtypes(include = ['object']).columns.tolist()
for col in cat_cols:
    value_counts = info[col].value_counts(normalize=True) * 100
    fig, ax = plt.subplots(figsize=(8, 3))
    #top_n = min(17, len(value_counts))
    #ax.barh(value_counts.index[:top_n], value_counts.values[:top_n])
    ax.barh(value_counts.index, value_counts.values)
    ax.set_xlabel('Percentage Distribution')
    ax.set_ylabel(f'{col}')
    plt.tight_layout()
    plt.show()
    j+=1
    # to show the str graphs of the first two distributions
    if j == 2:
        break

print("From the graphs above, we know that there is a perfect split of teams and sides. Let us first figure out the total amount data")
print("we have of team2 on terrorist side and the amount of times they have entered the light blue boundary")

# Code to find total amount of team2 on terrorist side games

total = 0
cur = 0

team = info[["team"]]
side = info[["side"]]

"""
while cur < 221330:
    if((team.loc[cur] == "Team2").bool() and (side.loc[cur] == "T").bool()):
        total+=1
    cur+=1

print(total)

boundary = 0
i = 0
while i < 221330:
    if((team.loc[i] == "Team2").bool() and (side.loc[i] == "T").bool()):
        if((boundaryZ.loc[i].astype('int') < 422).bool() or (boundaryZ.loc[i].astype('int') > 284).bool()):
            if((box.contains(Point(boundaryX.loc[i].astype('int'), boundaryY.loc[i].astype('int')))).bool()):
                boundary+=1
    i+=1

print(boundary)
"""



print("Boundary Total: 578")
print("Total Amount: 54910")

print("So, since we have the amount of times they were in the boundary and the total data of Team2 T side, we can find the probability")
print("by doing basic division.")
print("578/54910 is approximately 0.0105, which is around 1%.")
print("So, it is not a common strategy for this to happen because they have only done it one percent of their games.")

print("======================================================================================================================")

print("b. What is the average timer that Team2 on T (terrorist) side enters “BombsiteB” with least 2 rifles or SMGs?")

print("Let us first calculate the average equipment value for having at least 2 rifles or SMGs. Noting that a rifle is 2700 and ")
print("the lowest cost for an SMG is 1250, we can average this and say the average buy is around 4000 from the 'at least 2 rifles/smg'")
print("guns alone. Considering that the other teammates may buy armor/other utility, let us add 2000 to our approximation, finding the")
print("equipment value to be 6000.")

# This is the code ran to find the total Team2, terrorist and 2 smgs/rifles minimum and to find the total time in seconds
"""
time = 0
timer = info[["clock_time"]]
seconds = info[["seconds"]]
value = info[["equipment_value_freezetime_end"]]

# method to convert minutes to seconds
def get_seconds(clock_time):
    time_str = clock_time.values[0]  # Extract the string value from the pandas Series
    parts = time_str.split(':')  # Split the string on the ':' separator
    
    minutes = int(parts[0])  # Extract the minutes part as an integer
    seconds = int(parts[1])  # Extract the seconds part as an integer
    
    total_seconds = (minutes * 60) + seconds  # Calculate the total seconds
    return total_seconds
count = 0
i = 0
while i < 221330:
    if((team.loc[i] == "Team2").bool() and (side.loc[i] == "T").bool()):
        if((value.loc[i] > 6000).bool()):
            count+=1
            time+= int(get_seconds(timer.loc[i].astype(str)))
            if(int(get_seconds(timer.loc[i].astype(str))) == 0):
                time+=seconds.loc[i]
    i+=1
print(time, count)
"""

print("The calculated total time above is 173352 seconds")
print("Now, finding the average timer, we will have to divide total time in seconds by 3013, every Team2, terrorist, with (minimum)")
print("2 rifles/smgs from the dataset.")
print("The average time is: 57.53 seconds")

print("======================================================================================================================")

print("c. Now that we’ve gathered data on Team2 T side, let's examine their CT (counter-terrorist) Side. Using the same data set, tell our")
print("coaching staff where you suspect them to be waiting inside “BombsiteB”")
print("i. Hint: Try a heatmap")

area = info[["area_name"]]
# print(teamAndLocation.head(50))
unique_values = info['area_name'].unique().tolist()
"""
All possible locations
['TSpawn' 'TStairs' 'Tunnels' 'Fountain' 'LowerPark' 'Playground' 'Alley'
 'Connector' 'BombsiteA' 'Canal' 'Pipe' 'Water' 'Construction' 'UpperPark'
 'Restroom' 'Lobby' 'StorageRoom' 'SnipersNest' 'BackofA' 'Stairs'
 'UnderA' 'Walkway' 'Bridge' 'BombsiteB' None 'SideAlley']
"""
# above found by printing below
unique_values = info['area_name'].unique().tolist()
# array size 26
array = [0] * 26

i = 0
# run time approximately 40 seconds due to large amount of data
while i<221330:
    if((team.loc[i] == "Team2").bool() and (side.loc[i] == "CT").bool()):
        if(area.loc[i].isin(unique_values).bool()):
            index = unique_values.index(area.loc[i].item())
            array[index] += 1
    i+=1
print(array)
data = pd.DataFrame({
    'Location': ['TSpawn', 'TStairs', 'Tunnels', 'Fountain', 'LowerPark', 'Playground', 'Alley',
                'Connector', 'BombsiteA', 'Canal', 'Pipe', 'Water', 'Construction', 'UpperPark',
                'Restroom', 'Lobby', 'StorageRoom', 'SnipersNest', 'BackofA', 'Stairs',
                'UnderA', 'Walkway', 'Bridge', 'BombsiteB', None, 'SideAlley'],
    'Frequency': [array[0], array[1], array[2], array[3], array[4], array[5], array[6], array[7], array[8], 
                  array[9], array[10], array[11], array[12], array[13], array[14], array[15], array[16], array[17], 
                  array[18], array[19], array[20], array[21], array[22], array[23], array[24], array[25]]
})
# Reshape the data to create a matrix
heatmap_data = data.pivot(index=None, columns='Location', values='Frequency')

# Create the heatmap using seaborn
plt.figure(figsize=(15, 10))
sns.heatmap(heatmap_data,annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f', annot_kws={'color': 'black'})

# Add labels and title
plt.xlabel('Location')
plt.ylabel('Frequency')
plt.title('Heatmap of Frequency and Location')

# Show the plot
plt.show()


print("From the heatmap, shown in red, or the probability closer to the highest value (8447), will display which region most likely to  be in.")
print("So, from the heat map, I would tell our coaching staff that I'd suspect them to be waiting in LowerPark inside 'BombsiteB'")


# heat map (for fun) just for visualization purposes

num_cols = info.select_dtypes(include = ['int']).columns.tolist()
correlation_matrix = info[num_cols].corr()
plt.figure(figsize=(20, 15))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt='.2f')
plt.show()


print("======================================================================================================================")
print("3. (No Coding) Most of the time, our stakeholders (in this case, the CS:GO coaching staff) aren’t tech-savvy enough to run code")
print("themselves. Propose a solution to your product manager that:")

print("a. could allow our coaching staff to request or acquire the output themselves")
print("I would first propose that by transitioning to a probabilistic approach, the coaching staff will be able to acquire the output")
print("relatively easily, informing them of the team's playstyle and typical approach to the game, given their dataset. We can generalize it")
print("to each map and direct play style given that our code is generalizable. I would also preface that after creating each dataset, it is")
print("extremely easy to do acquire the output themselves, needing to run a few commands in the terminal.")

print("b. takes less than 1 weeks worth of work to implement")
print("As shown with this assessment, it is definitely possible to implement, but I would say that the only difficulty would be obtaining")
print("the dataset, as this is typically harder to obtain. Though, if many professional games are played and evaluated on each map, the dataset") 
print("can be created, but having a greater dataset will result in better probability of gamestyle and success. Though, in general, this can")
print("definitely be coded and implemented in a week.")

