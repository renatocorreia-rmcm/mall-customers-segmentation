import numpy as np
import pandas as pd

df = pd.read_csv("mall-customers.csv")

is_male = df["is_male"].values
income = df["annual-income"].values
score = df["spending-score"].values
age = df["age"].values

#todo: implement function to take df and return array of registers.