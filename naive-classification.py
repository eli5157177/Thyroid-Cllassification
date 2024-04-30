#%%
# Input file is numeric-no missing values: thyroid0387.csv 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('thyroid0387.csv')

df['age'] = df['age'].astype('float64')
for col in df.columns:
    if df[col].dtype == 'int64':
        df[col] = df[col].astype('category')
df.info()

#%% split data to train and test:
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
#stratied-split: (stratify=y) --> to keep the same distribution of classes
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']  



#%%
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.3,
                                                    random_state=0,
                                                    )
X_test.shape,X_train.shape





#%% Decision Tree Classifier with criterion gini index:
clf= GaussianNB()
#clf_GNB.fit(X_train, y_train)
clf.partial_fit(X_train, y_train, np.unique(y_train))

#%%


y_test_pred = clf.predict(X_test)
print('Model accuracy score naiive base: {0:0.4f}'. format(accuracy_score(y_test, y_test_pred)))


y_train_pred = clf.predict(X_train)
y_train_pred 


#%%
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_train_pred)))
print('Training set score: {:.4f}'.format(clf.score(X_train, y_train)))
print('Test set score: {:.4f}'.format(clf.score(X_test, y_test)))

#visualize the tree:

#%% confusiiion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# set color of the confusion matrix:


c_m = confusion_matrix(y_true=y_test, y_pred=y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=c_m,
                         display_labels= ['Negative','Positive'],
              
                         )
                         
disp.plot()
plt.show()



# %% Full report

from sklearn.metrics import classification_report

print(classification_report(y_test, y_test_pred,target_names=['Negative','Positive']))

# %%
