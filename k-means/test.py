""""""


""" 
	dependencies 
"""
import pandas as pd  # load dataframe
from kmeans import kmeans  # the algorithm
import matplotlib.pyplot as plt  # plot


""" 
	load dataframe
"""
df = pd.read_csv("../dataset/mall-customers.csv").drop('is_male', axis=1)


""" 
	run algorithm 
"""
centroids, clusters = kmeans(df=df, k=5, initializations=40)


""" 
	plot results 
"""
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(  # add datapoints
	df['age'].values, df['annual-income'].values, df['spending-score'].values, c=clusters, cmap='rainbow'
)
ax.scatter(  # add centroids
	centroids[:, 0], centroids[:, 1], centroids[:, 2], color="black", marker='D'
)
ax.set_xlabel('age'); ax.set_ylabel('annual-icome'); ax.set_zlabel('spending-score')  # set label for each axis
plt.show()
