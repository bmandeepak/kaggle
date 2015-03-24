__author__ = 'mandeepak'
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
sns.set_palette("deep", desat=.6)

def loadata(filename):
    data= pd.read_csv(filename,index_col='Id')
    return data.ix[:, :-1], data.ix[:, -1]

filename_train="train.csv"
train_data, train_labels = loadata(filename_train)

# Shape of data
print np.shape(train_data)

# Covariates
covariates = list(train_data.columns.values)

# Cross Validation to split the data into training and validation data set
train_X, validation_X, train_y, validation_y = cross_validation.train_test_split(train_data, train_labels) # splits 75%/25% by default


def getImportance(classifier):
    importances = classifier.feature_importances_
    std = np.std([tree.feature_importances_ for tree in classifier.estimators_],
             axis=0)
    indices=np.argsort(importances)[::-1][:10]
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature Importances")
    plt.bar(range(10), importances[indices],
       yerr=std[indices], align="center",alpha=.5)
    plt.xticks(range(10), np.asarray(covariates)[indices], rotation=45, rotation_mode="anchor", ha="right")
    plt.xlim([-1, 10])
    plt.show()
    return indices


# Base Case to use Random Forest
randf = RandomForestClassifier(n_estimators=100, bootstrap=True, oob_score=True)
randf.fit(train_X, train_y)
print "Initial Train csv score: %.2f" % randf.score(train_X, train_y)
predicted = randf.predict(validation_X)
print "F1 Score: %.2f" % metrics.f1_score(validation_y, predicted)

# Get Top 10 Features for Random Forest Classifier
top_indices = getImportance(randf)

def plot(train_X):

    train_data.ix[:,top_indices].hist(figsize=(16,10), bins=50)
    plt.show()

plot(train_X)