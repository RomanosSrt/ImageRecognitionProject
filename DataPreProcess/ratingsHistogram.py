import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

file_path = "ModifiedDataset/userNames.csv"

users = pd.read_csv(file_path)

bars = 20
n, bins, patches = plt.hist(users['count'], bins=bars, edgecolor='black')

# Normalize the heights to map them to a color range
cmap = cm.winter  # You can change to another colormap like 'plasma', 'inferno', etc.
colors = [cmap(i / bars) for i in range(bars)]  # Create a set of 15 colors

# Apply these colors to the bars
for i, patch in enumerate(patches):
    # Assign a color to each bar
    patch.set_facecolor(colors[i])
    
    
plt.yscale('log')
plt.ylabel('Users')
plt.xlabel('Ratings')
plt.title('Ratings per User Frequency Histogram')
plt.show()