import pandas as pd
from datetime import datetime
import numpy as np

df = pd.read_csv("Filpettrain.csv",index_col=0, parse_dates=True)

df.dtypes
df['Date'] = df.index.date
df['Time'] = df.index.time
df['Year'] = df.index.year
df['Month'] =df.index.month
df['day'] =df.index.day
df['hours'] =df.index.hour
df['minutes'] =df.index.minute
df['Weekday_Name'] =df.index.weekday_name
df['Day_of_Week'] =df.index.dayofweek

# monday = 0, sunday = 6

df['weekend'] = 0          # Initialize the column with default value of 0
df.loc[df['Day_of_Week'].isin([5, 6]), 'weekend'] = 1  # 5 and 6 correspond to Sat and Sun

print(df.tail(500))
df.to_csv('FP.csv')