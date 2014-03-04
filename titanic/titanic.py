###############################################################################
# http://www.kaggle.com/c/titanic-gettingStarted

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt#import pylab as pl
from scipy import linalg

from sklearn.linear_model import (LinearRegression, LassoCV, LassoLarsCV,
                                  Ridge, RidgeCV, ElasticNetCV,
                                  RandomizedLasso, lasso_stability_path)
from sklearn.feature_selection import (f_regression, SelectPercentile, RFECV,
                                       f_classif)
from sklearn.cross_validation import KFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import (Imputer, scale, StandardScaler,
                                   OneHotEncoder, LabelBinarizer, LabelEncoder)
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor

from itertools import combinations, combinations_with_replacement

import my_preprocessing as my

# scikitlearn breaks pandas installation: ignore the warning message
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        module="pandas", lineno=570)
warnings.filterwarnings("ignore", category=DeprecationWarning,
                        module="pandas", lineno=68)
warnings.filterwarnings("ignore", category=UserWarning,
                        module="sklearn", lineno=251)
warnings.filterwarnings("ignore", category=UserWarning,
                        module="sklearn", lineno=275)
# logit fit in R
# NOTE: Sex correlates perfectly with Title, so no need to include both
cols_signif_to_surv = ['Pclass', 'Sex', 'Age', 'SibSp', 'Pclass*Sex']
cols_insignif_to_surv = ['Name', 'Parch', 'Fare', 'Cabin', 'Embarked']

# lm fit in R
cols_signif_to_age = ['Pclass', 'Sex', 'SibSp', 'Cabin', 'Title']
cols_insignif_to_age = ['Name', 'Parch', 'Fare', 'Embarked']

# from the regressions above
cols_to_keep = [u'Survived',
                u'Pclass', u'Name', u'Sex', u'Age', u'SibSp', u'Cabin']

# load and merge
train = pd.read_csv('new_train.csv', index_col=0)
test = pd.read_csv('new_test.csv', index_col=0)
df = pd.concat([train, test])

###############################################################################
# missing values
# NOTE: Age will be custom-imputed later. Embarked will be binarized.
cols_to_impute = ['Fare']
#Imputer(strategy='median').fit_transform(df.Fare)
df.Fare = df.Fare.fillna(df.Fare.median())

# featurization: get Titles
cols_to_featurize = ['Name']
# split last and first names
names = np.char.split(df.Name.values.astype(str), ', ')
# keep first names only
names = [x[1] for x in names]
titles = my.get_all_titles(names)
# manually replace 'the'
# NOTE: 'Jonkheer' is a real title.
titles[759] = 'Countess.'
add_features = ['Title']
titles = pd.DataFrame({'Title':titles}, index=df.index)
df = df.join(titles)

# LabelBinarizer or Encoder
# NOTE: leave one out (avoid dummy trap/singular matrix)
#       'Title', 'Cabin' get binarized after featurization
# TODO: use df.apply() instead of for loop
cols_to_binarize = ['Sex', 'Embarked', 'Pclass']#, 'Title', 'Cabin']
for col in cols_to_binarize:
    df = df.join(pd.get_dummies(df[col], prefix=col).ix[:,1:])

# featurization: interaction
# NOTE: Pclass interacts with dummies 'Pclass_2' and 'Pclass_3'
cols_to_drop = ['Survived', 'Age']
df_to_featurize = df._get_numeric_data().drop(cols_to_drop,1)
interact = combinations(df_to_featurize.drop('Pclass',1), 2)
# combinations_with_replacement includes square
#interact = combinations_with_replacement(df_to_featurize.drop('Pclass',1), 2)
interact = [pd.DataFrame({':'.join((i,j)):df[i]*df[j]}) for i,j in interact]
df = df.join(interact)
# drop zero columns
df = df.drop(df.columns[(df == 0).all()],1)

# featurization
# NOTE: Age will be featurized after imputation
cols_to_featurize = ['Fare', 'Parch', 'Pclass', 'SibSp']
df_to_featurize = df[cols_to_featurize]
# featurization: log
# NOTE: log(x+1) to avoid log(0) for Fare, Parch and SibSp. Pclass never 0.
df = df.join(np.log(df_to_featurize+1), rsuffix='_log')
# featurization: 2nd degree poly
df = df.join(df_to_featurize**2, rsuffix='^2')

# LabelBinarizer or Encoder
cols_to_binarize = ['Title', 'Cabin']
for col in cols_to_binarize:
    df = df.join(pd.get_dummies(df[col], prefix=col).ix[:,1:])

# standardization
# NOTE: Age will be normalized after imputation
cols_to_normalize = ['Fare'] # actually, all numeric except Age and Survived
cols_to_drop = ['Survived', 'Age', 'Pclass']
df_to_scale = df._get_numeric_data().drop(cols_to_drop,1)
df[df_to_scale.columns] = scale(df_to_scale)

###############################################################################
# missing values
cols_to_impute = ['Age']
cols_to_drop.extend(['Cabin', 'Embarked', 'Name', 'Sex', 'Ticket', 'Title'])
# filter out features
def f(x):
    return True#':' not in x and '_log' not in x and '^2' not in x
X = df.drop(cols_to_drop,1).select(f,1)
Y = df.Age
# pandas fit
result = pd.ols(y=Y, x=X)
print result.r2

# sklearn fit with cross-validation
# mask missing values
i = df.Age.notnull(); y=Y[i]; x=X[i]

# Univariate feature selection with F-test for feature scoring
# We use the default selection function: the 10% most significant features
# http://scikit-learn.org/stable/modules/feature_selection.html
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(x, y)
# lm
lm = LinearRegression()
kfold = KFold(len(x), n_folds=20)
cvs = cross_val_score(lm, x, y, cv=kfold)
print cvs[cvs > 0].mean() #=0.394 #0.404 with selector.transform(x) (n=25)
# Lasso
lasso = LassoLarsCV()#cv=20)
cvs = cross_val_score(lasso, x, y, cv=kfold)
print cvs[cvs > 0].mean() #=0.427 #0.444 with all features (n=244)
# KNN
knn = KNeighborsRegressor()
n_neighbors = range(2,20,2)
params = dict(n_neighbors=n_neighbors)
grid_search = GridSearchCV(estimator=knn, param_grid=params).fit(x,y)
print grid_search.best_score_, grid_search.best_estimator_.get_params() #=0.3694
neigh = RadiusNeighborsRegressor(radius=1.0)
# Ridge
ridge = RidgeCV(cv=20)
cvs = cross_val_score(ridge, x, y, cv=kfold)
print cvs[cvs > 0].mean() #=0.434 #0.430 with all features (n=244)
# Elastic Net
elastic = ElasticNetCV()
cvs = cross_val_score(grid_search, x, y, cv=kfold)
print cvs[cvs > 0].mean() #=0.417
# SVM
svr = SVR()#kernel='rbf')
parameters = {'kernel':('linear', 'rbf'), 'C':np.logspace(-3,3,10)}
grid_search = GridSearchCV(svr, parameters).fit(x,y)
print grid_search.best_score_, grid_search.best_estimator_.get_params() #=.434

# feature selection
# http://scikit-learn.org/stable/auto_examples/plot_rfe_with_cross_validation.html
rfecv = RFECV(estimator=lasso).fit(x, y)

# impute
model = lasso#ridge#lm
model.fit(x, y)
i = df.Age.isnull()
#df.Age[i] = result.y_fitted[i]
df.Age[i] = model.predict(X[i])

# featurization
cols_to_featurize = ['Age']
age2 = pd.DataFrame({'Age^2':df.Age**2}, index=df.index)
log_age = pd.DataFrame({'log(Age)':np.log(df.Age)}, index=df.index)
df = df.join([age2, log_age])

# standardization
cols_to_normalize = ['Age', 'Age^2', 'log(Age)']
df[cols_to_normalize] = scale(df[cols_to_normalize])

###############################################################################
# classification
cols_to_predict = ['Survived']
cols_signif_to_surv = ['Pclass', 'Sex', 'Age', 'SibSp', 'Pclass*Sex']
cols_to_drop.pop(cols_to_drop.index('Age'))
# filter out features
def f(x):
    return True#':' not in x and '_log' not in x and '^2' not in x
X = df.drop(cols_to_drop,1).select(f,1)
Y = df.Survived
# train set
x = X.ix[train.index]
y = y.ix[train.index]

# feature selection
selector = SelectPercentile(f_classif, percentile=10).fit(x, y)
#scores = -np.log10(selector.pvalues_)
#scores = np.ma.masked_invalid(scores)
#scores /= scores.max()

# SVM
svc = SVC()
params = {'kernel':('linear', 'rbf'), 'C':np.logspace(-6,0,10),
          'gamma':np.logspace(-8, -3, 10)}
clf = GridSearchCV(estimator=svc, param_grid=params).fit(x, y)
print clf.best_score_, clf.best_estimator_.get_params()

# Trees
clf = RandomForestClassifier(n_estimators=200, max_depth=None,
                             min_samples_split=1, random_state=0)
scores = cross_val_score(clf, x, y)
scores.mean()

# Save predictions ready to submit
clf.fit(x,y)
x = X.ix[test.index]
output = np.c_[test.index.values, clf.predict(x)].astype(int)
fname = 'clf.svc.linear.x247'#'RandomForest200'
savetxt(fname + '.csv', output, fmt='%s', delimiter=',',
        header='PassengerId,Survived', comments='')

# Save to Matlab
data = np.c_[y.values, X.values]
fname = 'train'
matlab_name = '_'.join(('titanic', fname))
suffix = 'lm'
fname = '_'.join((matlab_name, 'x%d'%ndim, suffix))
scipy.io.savemat(save_dir + fname + '.mat', mdict={matlab_name: data})
