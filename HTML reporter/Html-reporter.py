#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# This script is part of the project: "Thyroid Disease Data Set"
# The script is used to clean and pre-process the data set
# The data set is available at: https://archive.ics.uci.edu/ml/datasets/thyroid+disease
# the data set is used for educational purposes only
# the script writen in python 3.10 
# The script uses the following libraries: pandas, numpy, matplotlib, seaborn, ydata_profiling
# The script is part of the project MAMAN 21 - Data Science course, Open University
# This script writen by :  Eli. B . 2024
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%%
import pandas as pd
from ydata_profiling import ProfileReport   # Importing the ProfileReport


# #%% take one: before imputing all data
# df = pd.read_csv("thyroid0387.csv")
# # %%
# profile = ProfileReport(df, title="Pandas Profiling Report")    
# profile.to_file("thyroid0387.html") # Saving the report to a file
# # %%
# %% take two: after imputing all data
# definning categorial for well reporting calc
# categorical auto detected.

df = pd.read_csv("thyroid0387.csv",)
profile = ProfileReport(df, title="THYROID Report")    
profile.to_file("thyroid0387.html") # Saving the report to a file
# %% take three: after imputing all data
df = pd.read_csv("thyroid0387.csv",)
profile = ProfileReport(df, title="THYROID Report")    
profile.to_file("..\\thyroid0387.html") # Saving the report to a file


# %%
