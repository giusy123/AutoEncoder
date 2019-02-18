# Importing pandas
import pandas as pd
import numpy as np
import os.path as path
import os
import pathlib


pathInput="KDDTest+"
pathOutput=pathInput+"aggregate"

# name last column mi serve per togliere la colonna classification dal dataframe
cls = ' classification.'

df = pd.read_csv(pathOutput + ".csv", index_col=0)
dt_Normal = df[(df[cls] == 'normal')]
dt_Dos = df[(df[cls] == 'Dos')]
dt_U2R = df[(df[cls] == 'U2R')]
dt_R2L = df[(df[cls] == 'R2L')]
dt_Probe = df[(df[cls] == 'Probe')]

print("Dos shape: ", dt_Dos.shape)
print("Normal shape: ", dt_Normal.shape)
print("Probe shape: ", dt_Probe.shape)
print("U2R shape: ", dt_U2R.shape)
print("R2L shape: ", dt_R2L.shape)
print(df[cls].unique())
print("All", df.shape)