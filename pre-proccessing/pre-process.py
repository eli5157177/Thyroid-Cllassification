
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
#%% Thyroid Disease Data Set preprocessing
# preprocessing steps included in the code below are:
# TODO:
# understand meaning of columns/read articles...................Done
# remove columns with no meaning................................Done
# set class attribute to 0,1....................................Done
# replace "?"" with np.nan......................................Done
# convert all columns to numeric................................Done
# deal with data types..........................................Done
# reproduce report..............................................Done
# add table of contents to the word document....................Done
# deal with missing values......................................Done
# clean outliers/mistakes.......................................Done
# encode categorical data.......................................Done
# save data.....................................................Done
# plot class distribution.......................................Done
# plot missing values...........................................Done
# plot numerical attributes.....................................Done
# plot categorical attributes...................................Done
# plot outliers.................................................Done
# plot correlations.............................................Done
# plot class distribution.......................................Done
#%% imports and settings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport   # Importing the ProfileReport
import seaborn as sns
types_dict = {
              'age': float, # age is int but will be converted to float later
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
              'fti': float,
              'tbg_measured': 'category',
              'tbg': float,
              'referral_source': 'category',
              'diagnosis': 'category'
              }

df = pd.read_csv("thyroid0387.csv", dtype=types_dict)
if not  df.empty: 
        df.info() 
else: 
       print('Data not loaded')
# set options for pandas:
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)

       
# %% numerical and categorical attributes
numAttr = df.select_dtypes(include=['float64']).columns.tolist()
catAttr = df.select_dtypes(include=['category']).columns.tolist()

# %% print info about the data

print( df.describe(include='all').T )
for i in df.columns:
    print('unique values in "{}":\n'.format(i),df[i].unique())
    
#%% produce a report
profile = ProfileReport(df, title="Pandas Profiling Report")    
profile.to_file("thyroid0387.html") # Saving the report to a file
# df.dtypes
 #%% plot missing values:

df.isna().sum().sort_values(ascending=True).plot(kind='bar', figsize=(10, 5),
                        title='Missing values per column')
# %% replace "?" with np.nan
df.replace('?', np.nan, inplace=True)
#%% analyze class column (run this cell only once)
def analyze_class(diagnosis):
        if diagnosis[0] == '-':      # no diagnosis negative class
            return 'No Diagnosis: Negative'
        if diagnosis[1] == '[':      # one letter diagnosis
            return (diagnosis[:1])
        if diagnosis[2] == '[':      # two letter diagnosis
            return diagnosis[:2]
        if diagnosis[3] == '[':
            return diagnosis[:3]    # three letter diagnosis
        return np.nan               # algo error - if np.nan will appear in the data
# build new diagnose column:
myClass = []
for i in df['diagnosis']:
    myClass.append( analyze_class(i))
print(myClass)

# %% plot class distribution
# optional : shows nice bar chart of class distribution

ax = df['diagnosis'].value_counts().plot(kind='barh',
                                       title='diagnosis distribution', figsize=(7, 9), 
                                       color='green', grid=True,
                                       fontsize=12, edgecolor='black', linewidth=2,
                                       position=0.5, width=1.0, label='diagnosis',
                                       legend=True, log=False,)
#adding text values to top of bars:
for rect in ax.patches:
    ax.text(rect.get_width(), rect.get_y() + rect.get_height() / 2,
            f'{rect.get_width():.0f}', ha='left', va='center')



#%% final classified as positive or negative

positives = ['A', 'B', 'C', 'D', 'E', 'F', 'G','H',
                 'I', 'J','L','M','N','O','P','Q',
                 'GI','OI','CI','MI','LJ','S',]

for val in df['class']:
    if val in positives:
        df['class'].replace(val, 'Positive', inplace=True)

recordant = ['R', 'DR','AK', 'GK', 'KJ', 'MK', 'FK','HK','GKJ']

for val in df['class']:
    if val in recordant:
        df['class'].replace(val, np.nan, inplace=True)
        
for val in df['class']:
    if val =='K':
        df['class'].replace(val, 'Negative', inplace=True)
#%% save data to csv files: classified + unclassified(for future use)
df = df.drop('diagnosis', axis=1)
dfRecordant = df[df['class'].isnull()]
df = df[df['class'].notnull()]
df.rename(columns={'class': 'diagnosis'}, inplace=True)
dfRecordant.rename(columns={'class': 'diagnosis'}, inplace=True)
dfRecordant.to_csv('un-classified.csv', index=False)
df.to_csv('throid0387.csv', index=False)

#%%

# %% numerical attributes distribution

plt.figure(figsize=(17, 15))
for col in df[numAttr]:
    ax = plt.subplot(4, 3, numAttr.index(str(str(col)))+1)
    sns.histplot(df[[col]],kde=True,  )
    plt.xlabel(str(col))
plt.tight_layout()
plt.show()

# %% numerical attributes boxplot
plt.figure(figsize=(17, 17))
for col in df[numAttr]:
    ax = plt.subplot(3, 3, numAttr.index(str(col)) + 1)
    sns.boxplot(x=df[col])
    plt.xlabel(str(col))
plt.tight_layout()
plt.show()

# %% correct age outliers:
#df.loc[df['age'] >99, 'age'] = np.nan

# %% one numeric boxplot

plt.figure(figsize=(2,2))
sns.boxplot(x=df['t3'])
plt.xlabel('t3')
plt.show()


# %% missing values print

df.isna().sum().sort_values(ascending=True).plot(kind='bar',
                                                 figsize=(10, 5),
                                                 title='Missing values per column')
# %% removing columns missing values/irrelevant
del df['tbg']   # tbg has more than 96% missing values
del df['fti']   # fti is a calculated value from tt4 and tbg
del df['measured_tsh']  # measured_tsh is a boolean value of tsh_measured for consistency
del df['measured_t3']   # measured_t3 is a boolean value of t3_measured for consistency
del df['measured_tt4']  # measured_tt4 is a boolean value of tt4_measured for consistency
del df['measured_t4u']  # measured_t4u is a boolean value of t4u_measured for consistency
del df['measured_tbg']  # measured_tbg is a boolean value of tbg_measured for consistency

# %%
#df['referral_source'].mode().values[0]
# %% drop records with more than half missing values
df.dropna(axis = 0, thresh = int(0.5*len(df.dlumns)), inplace = True)

# %% one numeric attribute boxplot

plt.figure(figsize=(2, 2))
sns.histplot(df[['t3']], kde=True)
plt.xlabel('t3')
plt.show()


# %% correct tsh outliers:
# ths query :loc(row_name,clumns_name) or iloc(intRow,intClumn)
# %% corolation matrix - numerical attributes

plt.figure(figsize=(15, 10))
sns.heatmap(df[numAttr].corr(method='pearson').abs(), annot=True
            , cmap='coolwarm' , vmin=0, vmax=1,
            square=True, linewidths=1,
            fmt='.2f', annot_kws={'size': 9},
             mask=np.triu(np.ones_like(df[numAttr].corr()))
             )
# %% linear regression - t3 vs t4u

sns.lmplot(x='t4u', y='tbg', data = df, line_kws={'color':'red'})
plt.title('TBG vs T4u - linear regression\n')


# %% categorical attributes distribution

plt.figure(figsize=(15, 17))
for col in df[catAttr]:
    ax = plt.subplot(5, 5, catAttr.index(str(col)) + 1)

    sns.countplot(df[catAttr], x=df[col], hue='diagnosis' ) #hue='class'
    plt.xlabel(str(col))
plt.tight_layout()
plt.show()



