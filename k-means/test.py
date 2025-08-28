import pandas as pd
import numpy as np
from kmeans import kmeans

df = pd.read_csv("dataset/mall-customers.csv")


centroids = kmeans(df, 5) 
print(centroids)

color = np.array(["red", "green", "blue", "purple", "cyan"])
point_colors = np.full(shape=200, fill_value="", dtype=object)
# make array of colors based on argmin dist(point, cluster)