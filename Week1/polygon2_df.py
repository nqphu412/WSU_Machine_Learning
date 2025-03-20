# -*- coding: utf-8 -*-


# It's customary to call pandas pd when importing it
import pandas as pd
polygons = {
    'Name': [
        'Triangle', 'Quadrilateral', 'Pentagon', 'Hexagon', 'Heptagon', 
        'Octagon', 'Nonagon', 'Decagon', 'Hendecagon', 'Dodecagon', 
        'Tridecagon', 'Tetradecagon'],
     # Range parameters are the start, the end of the range and the step
     'Sides': range(3, 15, 1),
}

polygons_data_frame = pd.DataFrame(polygons)

print(polygons_data_frame)

head5=polygons_data_frame.sort_values('Name').head(5)

print(head5)

polygons_data_frame[
   'Length of Name'
] = polygons_data_frame['Name'].str.len()

print(polygons_data_frame)   #str.len() similar to apply(len) as below

polygons_data_frame[
   'Length of Name'
] = polygons_data_frame['Name'].apply(len)

print(polygons_data_frame)

polygons_data_frame[
   'Length of Name'
] = polygons_data_frame['Name'].apply(lambda n: len(n))

print(polygons_data_frame)


# We use the DataFrame's plot method here, 
# where we specify that this is a scatter plot
# and also specify which columns to use for x and y
polygons_data_frame.plot(
    title='Sides vs Length of Name',
    kind='scatter',
    x='Sides',
    y='Length of Name',
)


## Question 3
values = []

for x in range(5):
    value = float(input("Enter 5 values"))
    values.append(value)

max(values)

import numpy as np ## Question 4
matrixA = np.random.randint(0,11, size = [2,3])
matrixA   ### 1

matrixB = np.random.randint(0,4, size = [3,4])
matrixB   ### 2

matrixA.T   ### 3

matrixA @ matrixB   ### 4


