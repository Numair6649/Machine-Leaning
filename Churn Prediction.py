import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

churn_df = pd.read_csv('churn.csv.txt')
churn_df.head()


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
churn_df['VMail Plan'] = le.fit_transform(churn_df['VMail Plan'].astype('str'))
churn_df.head()

churn_df['Churn?'] = le.fit_transform(churn_df['Churn?'].astype('str'))


churn_df['Int\' Plan'] = le.fit_transform(churn_df['Int\'l Plan'].astype('str'))

churn_df.head()

churn_df.columns

col = ['State','Area Code','Int\'l Plan','Phone']
churn_df = churn_df.drop(col,axis=1)
churn_df.head()

churn_df.dtypes

churn_df.plot(x = 'VMail Plan',y = 'Churn?',kind='scatter')

churn_df.plot(x = 'VMail Message',y = 'Churn?',kind='scatter')

churn_df.plot(x = 'CustServ Calls',y = 'Churn?',kind='scatter')

churn_df.plot(x = 'CustServ Calls',y = 'Churn?',kind='hist')

import sklearn
from sklearn.model_selection import train_test_split

X=churn_df.iloc[:,0:15]
Y=churn_df[['Churn?']]
X.head()

X_train,X_test,Y_train,Y_test=train_test_split(X,Y)

X_train.shape

X_train.tail()

Y_train.head()

type(Y_train)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)


knn.fit(X_train, Y_train)

knn.score(X_test, Y_test)

from sklearn import svm
clf = svm.SVC(kernel='linear', C=1).fit(X_train, Y_train)

clf.score(X_test, Y_test)


