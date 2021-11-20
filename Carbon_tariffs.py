from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime

# Import data
EU_ETS = pd.read_csv("Data/EUETS/EUETSPrices.csv")

# Average price of the last month

EU_ETS["Date"] = pd.to_datetime(EU_ETS["Date"]).dt.date
ETS_mean_month = EU_ETS.loc[EU_ETS["Date"] > pd.to_datetime("2021-10-19")][
    "Price"
].mean()
