__author__ = 'mandeepak'
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import ShuffleSplit, train_test_split, Bootstrap
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import Imputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.learning_curve import learning_curve
from sklearn import cross_validation
sns.set_palette("deep", desat=.6)

train_data = pd.read_csv("train.csv",index_col='Id')
test_data = pd.read_csv("test.csv",index_col='Id')

# Shape of data
print np.shape(train_data)

# Covariates
covariates = list(train_data.columns.values)

# Cross Validation to split the data into training and validation data set
#train_X, validation_X, train_y, validation_y = cross_validation.train_test_split(train_data, train_labels) # splits 75%/25% by default


def getImportance(classifier,covariates):
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
    #plt.show()
    return indices


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def crossvalidation(estimator,params,jobs):
    cv = ShuffleSplit(X_train.shape[0], n_iter=2, test_size=0.2)
    classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=params,n_jobs=jobs,scoring="accuracy")
    classifier.fit(X_train, y_train)
    print "Best Estimator using Grid Search"
    print classifier.best_estimator_
    return cv, classifier.best_estimator_


# Base Case to use Random Forest
randf = RandomForestClassifier(n_estimators=100, bootstrap=True, oob_score=True)
randf.fit(train_data.ix[:,:-1].values, train_data.ix[:,-1].values.ravel())
print "Initial Train csv score: %.2f" % randf.score(train_data.ix[:,:-1].values, train_data.ix[:,-1].values.ravel())

# Get Top 10 Features for Random Forest Classifier
top_indices = getImportance(randf,list(train_data.columns.values))

def plot(train_X,top_indices):
    train_data.ix[:,top_indices].hist(figsize=(15,10), bins=50)
    plt.show()

#plot(train_data,top_indices)

# Feature Engineering

# Mappping Horizontal_Distance_To_Hydrology  & Vertical_Distance_To_Hydrology  to a diagonal distance


temp=train_data.copy()
cols=temp.columns.tolist()
cols=cols[:8]+cols[9:]+[cols[8]]
temp=temp[cols]
del temp['Cover_Type']

X,y,X_train_data_missing,y_train_data_missing= temp[temp.Hillshade_3pm!=0].values[:,:-1],temp[temp.Hillshade_3pm!=0].values[:,-1:].ravel(),temp[temp.Hillshade_3pm==0].values[:,:-1],temp[temp.Hillshade_3pm==0].values[:,-1:].ravel()


X_train,X_test,y_train,y_test=train_test_split(X,y)
imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(X_train,y_train)
temp.Hillshade_3pm.loc[temp.Hillshade_3pm==0]=imp.transform(X_train_data_missing)
train_data.Hillshade_3pm=temp.Hillshade_3pm


temp=test_data.copy()
cols=temp.columns.tolist()
cols=cols[:8]+cols[9:]+[cols[8]]
temp=temp[cols]



X_test_data_missing= temp[temp.Hillshade_3pm==0].values[:,:-1]
temp.Hillshade_3pm.loc[temp.Hillshade_3pm==0]=imp.transform(X_test_data_missing)
test_data.Hillshade_3pm=temp.Hillshade_3pm






train_data['hypdis_hydro'] = np.sqrt(np.square(train_data.Vertical_Distance_To_Hydrology)+np.square(train_data.Horizontal_Distance_To_Hydrology))
train_data.slope_hyd=train_data.hypdis_hydro.map(lambda x: 0 if np.isinf(x) else x)
test_data['hypdis_hydro'] = np.sqrt(np.square(test_data.Vertical_Distance_To_Hydrology)+np.square(test_data.Horizontal_Distance_To_Hydrology))
test_data.hypdis_hydro=test_data.hypdis_hydro.map(lambda x: 0 if np.isinf(x) else x)


train_data['water_source']=1*np.array(train_data.Vertical_Distance_To_Hydrology>0)
train_data.water_source=train_data.water_source.map(lambda x: 0 if np.isinf(x) else x)
test_data['water_source']=1*np.array(test_data.Vertical_Distance_To_Hydrology>0)
test_data.water_source=test_data.water_source.map(lambda x: 0 if np.isinf(x) else x)

train_data['Mean_Fire_Hydrology_Road']=np.mean(train_data.Horizontal_Distance_To_Fire_Points + train_data.Horizontal_Distance_To_Hydrology + train_data.Horizontal_Distance_To_Roadways)
train_data.Mean_Fire_Hydrology_Road=train_data.Mean_Fire_Hydrology_Road.map(lambda x: 0 if np.isinf(x) else x)
test_data['Mean_Fire_Hydrology_Road']=np.mean(test_data.Horizontal_Distance_To_Fire_Points + test_data.Horizontal_Distance_To_Hydrology + test_data.Horizontal_Distance_To_Roadways)
test_data.Mean_Fire_Hydrology_Road=test_data.Mean_Fire_Hydrology_Road.map(lambda x: 0 if np.isinf(x) else x)


train_data['Mean_Fire_Hydrology']=np.mean(train_data.Horizontal_Distance_To_Fire_Points + train_data.Horizontal_Distance_To_Hydrology )
train_data.Mean_Fire_Hydrology=train_data.Mean_Fire_Hydrology.map(lambda x: 0 if np.isinf(x) else x)
test_data['Mean_Fire_Hydrology']=np.mean(test_data.Horizontal_Distance_To_Fire_Points + test_data.Horizontal_Distance_To_Hydrology )
test_data.Mean_Fire_Hydrology=test_data.Mean_Fire_Hydrology.map(lambda x: 0 if np.isinf(x) else x)

train_data['Mean_Fire_Roadways']=np.mean(train_data.Horizontal_Distance_To_Fire_Points + train_data.Horizontal_Distance_To_Roadways )
train_data.Mean_Fire_Roadways=train_data.Mean_Fire_Roadways.map(lambda x: 0 if np.isinf(x) else x)
test_data['Mean_Fire_Roadways']=np.mean(test_data.Horizontal_Distance_To_Fire_Points + test_data.Horizontal_Distance_To_Roadways )
test_data.Mean_Fire_Roadways=test_data.Mean_Fire_Roadways.map(lambda x: 0 if np.isinf(x) else x)

# Covariates
covariates_train = list(train_data.columns.values)
covariates_test = list(test_data.columns.values)

cols_train=train_data.columns.tolist()
cols_train=cols_train[:10]+cols_train[-5:]+cols_train[10:-5:]
train_data=train_data[cols_train]

cols_test=test_data.columns.tolist()
cols_test=cols_test[:10]+cols_test[-5:]+cols_test[10:-5:]
test_data=test_data[cols_test]

X_train, X_test, y_train, y_test = train_test_split(train_data.ix[:,:-1].values, train_data.ix[:,-1].values.ravel(),test_size=0.1)
print X_train.shape, X_test.shape, y_train.shape, y_test.shape



# randf = RandomForestClassifier(n_estimators=100, bootstrap=True, oob_score=True)
# randf.fit(X_train,y_train)
# print "Initial Train csv score: %.2f" % randf.score(X_train,y_train)


estimator = RandomForestClassifier()

params_grid={'n_estimators':[500],
                            'max_depth':[8,10]
            }
cv,best_estimator=crossvalidation(estimator, params_grid, 1)
print "Best Estimator Parameters"
#
print "n_estimators: %d" %best_estimator.n_estimators
print "Training Score(F1): %.2f" %best_estimator.score(X_train,y_train)


print getImportance(best_estimator, list(train_data.columns.values))

title = "Learning Curves (Random Forests, n_estimators=%d, max_depth=%.6f)" %(best_estimator.n_estimators,  best_estimator.max_depth)
plot_learning_curve(best_estimator, title, X_train, y_train, cv=cv, n_jobs=1)
plt.show()

y_pred=best_estimator.predict(X_test)
print "Training Score: %.2f" %best_estimator.score(X_train,y_train)
print "Test Score: %.2f" %best_estimator.score(X_test,y_test)
print "Classification Report - Test"
print metrics.classification_report(y_test, y_pred)

temp=test_data.copy()
temp['Cover_Type']=best_estimator.predict(temp.values)
temp=temp['Cover_Type']
temp.to_csv('RF-FeatureEng.csv', header=True)


class_weights=pd.DataFrame({'Class_Count':temp.groupby(temp).agg(len)}, index=None)
print class_weights
class_weights['Class_Weights'] = temp.groupby(temp).agg(len)/len(temp)
print class_weights

sample_weights=class_weights.ix[y_train]
print sample_weights

best_estimator.fit(X_train, y_train, sample_weight=sample_weights.Class_Weights.values)

temp=test_data.copy()
temp['Cover_Type']=best_estimator.predict(temp.values)
temp=temp['Cover_Type']
temp.to_csv('RF-FeatureEng_Weight.csv', header=True)


