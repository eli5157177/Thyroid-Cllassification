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
# input file is thyroid0387_numeric.csv
# output file is thyroid0387-Iterative-imputed.csv

import pandas as pd

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor

# Define the data types in the dataset - 
# this is important especially for categorical data!
# if not defined, pandas will treat them as float64
# and this will cause problems in the imputation step.
types_dict = {
              'age': float,
              'sex': 'category',
              'thyroxine': 'category',
              'query_thyroxine': 'category',
              'antithyroid_med': 'category',
              'sick': 'category',
              'pregnant': 'category',
              'surgery': 'category',
              'i131': 'category',
              'query_hypothyroid':'category',
              'query_hyperthyroid': 'category',
              'lithium': 'category',
              'goitre': 'category',
              'tumor': 'category',
              'hypopituitary': 'category',
              'psych': 'category',
              'tsh_measured': 'category',
              'tsh': float,
              't3_measured': 'category',
              't3': float,
              'tt4_measured': 'category',
              'tt4': float,
              't4u_measured': 'category',
              't4u': float,
              'fti_measured': 'category',
              #'fti': float,
              'tbg_measured': 'category',
              #'tbg': float,
              'referral_source': 'category',
              'diagnosis': 'category'
              }
# Load dataset
df=pd.read_csv('thyroid0387_numeric.csv', dtype=types_dict)
df.info()
#%% impute missing values
imputer = IterativeImputer(    estimator=DecisionTreeRegressor(),
                               max_iter=10, random_state=42)
#%% impute missing values
dfi=imputer.fit_transform(df) # impute missing values

# %%
dfi=pd.DataFrame(dfi, columns=df.columns)
dfi.info()
# %%
dfi.to_csv('thyroid0387-Iterative-imputed.csv', index=False)
