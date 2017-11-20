import numpy as np
import pandas as pd
import matplotlib as plot
from scipy import sparse
from sklearn import preprocessing
data = pd.read_csv("c:/data/train.csv")
data.Age.plot(kind="hist")


