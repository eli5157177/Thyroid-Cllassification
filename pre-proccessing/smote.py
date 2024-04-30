
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
# File to implement SMOTE algorithm to balance the dataset
#%%
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE


df = pd.read_csv('..\\thyroid0387.csv')
df.info()


# %% before SMOTE
df['diagnosis'].value_counts().plot(kind='pie',
                                autopct='%1.1f%%'
                                , labels=['Negative', 'Positive']
                                )

# %%
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

resampled_data = SMOTE().fit_resample(X, y)
X = resampled_data[0]
y = resampled_data[1]
#%%
y.value_counts().plot(kind='pie',
                                autopct='%1.1f%%'
                                , labels=['Negative', 'Positive']
                                )

# %%
df.info()
# %%