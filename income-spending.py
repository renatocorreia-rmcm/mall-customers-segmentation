import pandas as pd
import matplotlib.pyplot as plt


# GETTING VALUES

df = pd.read_csv("Mall_Customers.csv")
income = df["annual-income"].values
score = df["spending-score"].values


# PLOTTING

figure, ax = plt.subplots()
ax.scatter(income, score)


# ADDING TEXT

ax.set_title("Monetary")
ax.set_ylabel("spending score")
ax.set_xlabel("annual income (kU$)")


# SHOW

plt.show()
