import pandas as pd
from kmeans import kmeans

df = pd.read_csv("dataset/mall-customers.csv")


centroids = kmeans(df, 5) 
print(centroids)