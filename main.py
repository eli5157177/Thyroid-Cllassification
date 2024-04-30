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

#****************************************************************************************************
#                               Libraries and Data
#****************************************************************************************************
#%% load libraries and data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve   # Receiver Operating Characteristic
from sklearn.metrics import roc_auc_score
from plotly import graph_objects as go
from plotly.subplots import make_subplots

# chang options to display all columns:
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 50)


# Note: the data is already preprocessed and cleaned
# All of the 4 files are the same data set but with different preprocessing
# load data for tree decision classifiers
# first data set is without SMOTE and the second is with SMOTE
df = pd.read_csv('thyroid0387.csv')
df_SMOTE = pd.read_csv('thyroid0387_SMOTE.csv')
 
#****************************************************************************************************
#                               functions section
#****************************************************************************************************
#%%  function splits data to train and test:
def split_data(df):
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']  #target
    #stratied-split: (stratify=y) --> to keep the same distribution of classes
    X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                        test_size=0.3,
                                                        random_state=42,
                                                        stratify=y,
                                                        )
    return X_train, X_test, y_train, y_test

#print( X_test.shape,X_train.shape ) # show the shape of the data after split
#----------------------------------------------------------------------------------------------------
# Function to encode the data using one-hot encoding:
def myHotEncoder(x):
    #Extract categorical columns from the dataframe
    #Here we extract the columns with object datatype as they are the categorical columns
    catCols = x.select_dtypes(include=['int']).columns.tolist()
    #Initialize OneHotEncoder to output pandas dataframe
    encoder = OneHotEncoder(sparse_output=False)
    
    # Apply one-hot encoding to the categorical columns
    encoded = encoder.fit_transform(x[catCols])
    
    #Create a DataFrame with the one-hot encoded columns
    #We use get_feature_names_out() to get the column names for the encoded data
    xdf = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(catCols))
    
    # Concatenate the one-hot encoded dataframe with the original dataframe
    df_encoded = pd.concat([x.reset_index(drop=True), xdf.reset_index(drop=True)], axis=1)
   
    # Drop the original categorical columns

    return df_encoded.drop(catCols, axis=1)
#----------------------------------------------------------------------------------------------------
# a fuction for training the model:
def train_model(X_train, X_test, y_train ,modelnum):
    
    # Decision Tree Classifier 0- without SMOTE, 1- with SMOTE
    if modelnum == 0 or modelnum == 1: 
        clf = DecisionTreeClassifier(criterion='gini',
                                     max_depth=3,
                                     random_state=7,
                                     min_samples_leaf=25,  # if make it 1, the model will be overfitting
                                ) 
    
        clf.fit( X_train, y_train) 
        
    # Naive Bayes Classifier 2- without SMOTE, 3- with SMOTE   
    elif modelnum == 2 or modelnum == 3:
    
        clf= GaussianNB()

        clf.fit(X_train, y_train) 
                
    y_test_pred = clf.predict(X_test)
    y_train_pred = clf.predict(X_train)
    

    return clf, y_test_pred, y_train_pred
#----------------------------------------------------------------------------------------------------
#Model accuracy train and test:
def print_accuracy(y_train, y_test, y_train_pred, y_test_pred):
    print('Training accuracy({} samples) Accuracy: {}'. format(
                                                len(y_train),
                                                accuracy_score(y_true = y_train, y_pred = y_train_pred)))

    print ('Test Accuracy({} samples) Accuracy: {}'.format(
                                                len(y_test),
                                                accuracy_score(y_true = y_test, y_pred = y_test_pred)))    
    print('\n\n')
    # equivalent code : 
    #   print('Test set score: {}'.format(clf.score(X_test, y_test)))
    #   print('Test set score: {}'.format(clf.score(X_train, y_train)))
    

    print(classification_report(y_true=y_test,
                                y_pred=y_test_pred,
                                target_names=['Negative','Positive']))

 # function to print confusion matrix  
def print_confusion_matrix(y_test, y_test_pred):
    c_m = confusion_matrix(y_true=y_test, y_pred=y_test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=c_m,
                         display_labels= ['Negative','Positive'],)
    disp.plot()
    plt.show()
#----------------------------------------------------------------------------------------------------
#visualize the tree:
def visualize_tree(clf, X_train, y_train):
    plt.figure(figsize=(18,14))
    tree.plot_tree(clf.fit(X_train, y_train), feature_names=X_train.columns.to_list(), 
               class_names=['Negative','Positive'], label='all', 
               filled=True, precision=1,proportion=False,
               fontsize=12,rounded=True, node_ids=True,
               )
    plt.show()
#----------------------------------------------------------------------------------------------------
#****************************************************************************************************
#                                        MAIN CODE:
#****************************************************************************************************
  #%%  1- Decision Tree Classifier
modelNO = 0 # 0- Decision Tree Classifier, 1- Decision Tree Classifier with SMOTE
X_train, X_test, y_train, y_test = split_data(df)
clf, y_test_pred, y_train_pred = train_model(X_train, X_test, y_train, modelNO)
print_accuracy(y_train, y_test, y_train_pred, y_test_pred)
#print_confusion_matrix(y_test, y_test_pred)
#visualize_tree(clf, X_train, y_train)

# roc curve information for usind in the roc curve plot
prob = pd.DataFrame(clf.predict_proba(X_test))

# using roc_curve and roc_auc_score to get the roc curve and auc score
# for each model (0-3) roc contains fpr, tpr, thresholds, auc contains auc score
# for each model the data is stored in roc0, auc0, roc1, auc1, roc2, auc2, roc3, auc3
roc0 = roc_curve(y_test, prob[1])
auc0 = roc_auc_score(y_test, prob[1])
#****************************************************************************************************

# %% 2- Decision Tree Classifier with SMOTE
modelNO = 1
model_str = 'Decision Tree - SMOTE'
X_train, X_test, y_train, y_test = split_data(df_SMOTE)
clf, y_test_pred, y_train_pred= train_model(X_train, X_test, y_train, modelNO)
                                            
print_accuracy(y_train, y_test, y_train_pred, y_test_pred)
#print_confusion_matrix(y_test, y_test_pred)
#visualize_tree(clf, X_train, y_train)

prob = pd.DataFrame(clf.predict_proba(X_test))
# using roc_curve and roc_auc_score to get the roc curve and auc score
# for each model (0-3) roc contains fpr, tpr, thresholds, auc contains auc score
# for each model the data is stored in roc0, auc0, roc1, auc1, roc2, auc2, roc3, auc3

roc1 = roc_curve(y_test, prob[1])
auc1 = roc_auc_score(y_test, prob[1])

#****************************************************************************************************
# %% 3- Naive Bayes Classifier, one hot encoding
modelNO = 2
X_train, X_test, y_train, y_test = split_data(df)
# one-hot encoding:
X_train = myHotEncoder(X_train)
X_test = myHotEncoder(X_test)

clf, y_test_pred, y_train_pred= train_model( X_train, X_test, y_train, modelNO)
print_accuracy(y_train, y_test, y_train_pred, y_test_pred)
print_confusion_matrix(y_test, y_test_pred)
# using roc_curve and roc_auc_score to get the roc curve and auc score
# for each model (0-3) roc contains fpr, tpr, thresholds, auc contains auc score
# for each model the data is stored in roc0, auc0, roc1, auc1, roc2, auc2, roc3, auc3
prob = pd.DataFrame(clf.predict_proba(X_test))
roc2 = roc_curve(y_test, prob[1])
auc2 = roc_auc_score(y_test, prob[1])

#****************************************************************************************************
# %% 4- Naive Bayes Classifier with SMOTE
modelNO = 3
X_train, X_test, y_train, y_test = split_data(df_SMOTE)

# one-hot encoding:
X_test = myHotEncoder(X_test)
X_train = myHotEncoder(X_train)

clf, y_test_pred, y_train_pred = train_model(X_train, X_test, y_train,modelNO)
print_accuracy(y_train, y_test, y_train_pred, y_test_pred)
print_confusion_matrix(y_test, y_test_pred)
# using roc_curve and roc_auc_score to get the roc curve and auc score
# for each model (0-3) roc contains fpr, tpr, thresholds, auc contains auc score
# for each model the data is stored in roc0, auc0, roc1, auc1, roc2, auc2, roc3, auc3
prob = pd.DataFrame(clf.predict_proba(X_test))
roc3 = roc_curve(y_test, prob[1])
auc3 = roc_auc_score(y_test, prob[1])
#****************************************************************************************************
# %% print roc curves


#roc(model number)[0] = fpr
#roc(model number)[1] = tpr
  
fig = go.Figure()
fig.add_trace(  go.Scatter(x=roc0[0], y=roc0[1],
                mode='lines',
                name=f'AUC-decision Tree: {round(auc0,3)}')
                ) 
                  
                    
fig.add_trace(go.Scatter(x=roc1[0], y=roc1[1],
                        mode='lines',
                        name=f'AUC-decision Tree (with SMOTE): {round(1,3)}')
                    )

fig.add_trace(go.Scatter(x=roc2[0], y=roc2[1],
                    mode='lines',
                    name=f'AUC-Naive Base: {round(auc2,3)}')
              )
                    
fig.add_trace(go.Scatter(x=roc3[0], y=roc3[1],
                        mode='lines',
                        name=f'AUC-Naive Base (with SMOTE): {round(auc3,3)}')
              )
    
fig.update_layout( title="ROC Curve - AUC : Area Under the Curve",
                       
                      xaxis_title="FPR", 
                      yaxis_title="TPR",)

fig.update_yaxes(scaleanchor="y", scaleratio=700)
fig.update_xaxes(constrain='domain')
fig.add_shape(
    type='line', line=dict(dash='dot', color='black', width=2),
    x0=0, x1=1, y0=0, y1=1)
fig.show()


# %%
