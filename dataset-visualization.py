import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# LOAD AXIS

df = pd.read_csv("mall-customers.csv")

is_male = df["is_male"].values
income = df["annual-income"].values
score = df["spending-score"].values
age = df["age"].values


# PLOT

fig = plt.figure()  # empty figure without axes
ax = fig.add_subplot(projection='3d')

ax.set_xlabel('annual-income')
ax.set_ylabel('spending-score')
ax.set_zlabel('age')

colors = np.full(is_male.shape, "blue", dtype=object)
colors[is_male == 0] = "crimson"

ax.scatter(income, score, age, color=colors)
ax.set_title("Whole dataset")

# SHOW
plt.show()
