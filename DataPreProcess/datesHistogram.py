import pandas as pd
import ast
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib import cm

file_path = "OldMods/userNames.csv"

users = pd.read_csv(file_path)
users['date'] = users['date'].apply(ast.literal_eval)
all_dates = [date for sublist in users['date'] for date in sublist]

def y_axis_formatter(y, pos):
    return f'{int(y/1000)}k'


bars = 20
plt.hist(all_dates, bins=bars, edgecolor='black')
plt.gca().yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))
plt.ylabel('Ratings')
plt.xlabel('Year')
plt.title('Ratings per Year Frequency Histogram')
plt.show()