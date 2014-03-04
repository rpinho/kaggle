from numpy import *
import scipy.io
from itertools import product
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.metrics import mean_squared_error
import pandas as pd

save_dir = '' #/Users/ricardo/coursera/ml-003/ex6-003/mlclass-ex6/'

FEATURES = ['survived', 'pclass', 'last name', 'title + other names', 'sex',
            'age', 'sibsp', 'parch', 'ticket', 'fare', 'cabin', 'embarked']
CITIES = ['C', 'S', 'Q']
ALL_TITLES0 = ['Capt.', 'Col.', 'Don.', 'Dr.', 'Lady.', 'Major.', 'Master.',
               'Miss.', 'Mlle.', 'Mme.', 'Mr.', 'Mrs.', 'Ms.', 'Rev.', 'Sir.']
ALL_TITLES1 = ['Col.', 'Dr.', 'Major.', 'Master.', 'Miss.', 'Mlle.', 'Mr.',
               'Mrs.', 'Rev.']
ALL_TITLES2 = ['Dr.', 'Master.', 'Miss.', 'Mr.', 'Mrs.', 'Rev.']
CUSTOM_TITLES = ['Col.', 'Master.', 'Miss.', 'Mr.', 'Mrs.', 'Rev.']
FEATURIZE = {3:CUSTOM_TITLES}
IMPUTE = {5:'titles_sibsp_parch_median',#'titles_median',
          6:None, 7:None, 9:'mean', 11:'mean'}
PREPROCESSING = dict(FEATURIZE.items() + IMPUTE.items())
NORMALIZE = 'all'#[3,6]
MIN_N_TITLES = 1

def mask_missing_data(data):
    data[data == ''] = nan
    return ma.masked_invalid(data.astype(float))

def imputer(data, strategy):
    # data is MaskedArray
    if strategy == 'mean':
        return ma.mean(data)
    if strategy == 'median':
        return ma.median(data)
    if strategy == 'NaN':
        return nan

def featureNormalize(data, iNormalize):
    # location
    mu = data.mean(axis=0)
    # scale
    sigma = data.std(axis=0)
    #i_nonzero = sigma.nonzero()[0]
    X_norm = data.copy()
    X_norm[:,iNormalize] -= mu[:,iNormalize]
    X_norm[:,iNormalize] /= sigma[:,iNormalize]
    return X_norm

def get_family_members_matrix(data):
    return array([data == x for x in unique(data)])

def edit_some_names(data, min_n):
    # TODO: incomplete
    # Decided to manually edit the list (5 entries in train.csv, 0 in test.csv)
    name = data.T[3]
    unique_titles = get_unique_titles(name, min_n)
    # remove dot ('.') and add space at the end of title
    unique_titles = [x.split('.')[0] + ' ' for x in unique_titles]

def get_all_titles(name):
    split_names = char.split(name)
    return [x[0] for x in split_names]

def get_unique_titles(name, min_n):
    all_titles = get_all_titles(name)
    unique_titles = unique(all_titles)
    n_titles = [char.count(all_titles, x) for x in unique_titles]
    i = array([nonzero(x)[0].size > min_n for x in n_titles])
    return unique_titles[i]

def get_title_features(names, min_n, titles=[]):
    if not list(titles): titles = get_unique_titles(names, min_n)
    return array([char.find(names, title) != -1 for title in titles])

def get_titles_mean_ages(data, min_n, statistic):
    title_mx = get_title_features(data.T[3], min_n)
    age = mask_missing_data(data.T[5])
    return array([statistic(age[x]) for x in title_mx])

def get_missing_ages(data, min_n, statistic):
    #shape = (9, 891)
    title_mx = get_title_features(data.T[3], min_n)
    #shape = (9,)
    title_mean_age = get_titles_mean_ages(data, min_n, statistic)
    #shape = (891,)
    return dot(title_mx.T, title_mean_age)

def get_mx_relevant_for_age(data, min_n):
    # shape = (9, 891)
    title_mx = get_title_features(data.T[3], min_n)
    # shape = (7, 891)
    sibsp_mx = get_family_members_matrix(data.T[6])
    # shape = (7, 891)
    parch_mx = get_family_members_matrix(data.T[7])
    # order the matrices for the average and the sum product
    return title_mx, sibsp_mx, parch_mx

def get_avg_age_matrix(data, min_n, statistic):
    # boolean matrices
    mx = get_mx_relevant_for_age(data, min_n)
    mx_shape = [x.shape[0] for x in mx]
    # shape = (891,)
    age = mask_missing_data(data.T[5])
    # shape = (9, 7, 7)
    avg_age_mx = [statistic(age[x & y & z]) for x, y, z in product(*mx)]
    avg_age_mx = ma.masked_invalid(avg_age_mx)
    return avg_age_mx.reshape(mx_shape)

def get_missing_ages2(data, min_n, statistic):
    mx = get_mx_relevant_for_age(data, min_n)[::-1]
    avg_age_mx = get_avg_age_matrix(data, min_n, statistic)
    # sum product
    a = avg_age_mx.filled(0)
    b = einsum('ijk,kl->ijl', a, mx[0])
    c = einsum('ijl,jl->il',  b, mx[1])
    d = einsum('il, il->l',   c, mx[2])
    return d

def parse_data(data, preprocessing, min_n):
    _delete = [] # columns to delete
    _append = [] # columns to append

    # survived
    j = 0

    # pclass
    j = 1

    # last name
    j = 2
    _delete.append(j)

    # title + other names
    j = 3
    _delete.append(j)
    unique_titles = preprocessing[j]
    if unique_titles:
        name = data.T[j]
        _append = get_title_features(name, min_n, unique_titles).T.astype(int)

    # sex: map male = 1, female = 0
    j = 4
    sex = data.T[j]
    data[:,j] = (sex == 'male').astype(int)

    # age
    # TODO: linear regression
    #       **.6 ?
    # age.mask.nonzero()[0].size = 177 (177 missing values)
    j = 5
    strategy = preprocessing[j]
    if strategy == 'delete':
        _delete.append(j)
    else:
        name = data.T[3]
        age = mask_missing_data(data.T[j])

        # mean or median
        if 'titles' in strategy:
            statistic = getattr(ma, strategy.split('_')[-1])

        # estimates based on title and number of family numbers
        if 'titles_sibsp_parch' in strategy:
            # estimates based on title only
            age_from_title = get_missing_ages(data, min_n, statistic)
            # estimates based on title and number of family numbers
            age_from_title_sibsp_parch = get_missing_ages2(data, min_n, statistic)
            # some estimates are missing
            i_zero = age_from_title_sibsp_parch == 0
            # replace missing with title-only estimate
            age_from_title_sibsp_parch[i_zero] = age_from_title[i_zero]
            # finally impute the missing age in the data matrix
            data[age.mask,j] = age_from_title_sibsp_parch[age.mask]

        # estimates based on title only
        elif 'titles' in strategy:
            data[age.mask,j] = get_missing_ages(data, min_n, statistic)[age.mask]

        # one estimate only: mean, median or NaN
        else:
            data[age.mask,j] = imputer(age, strategy)

    # sibsp
    j = 6
    if preprocessing[j] == 'delete':
        _delete.append(j)
    elif preprocessing[j] == 'binary':
        i_nonzero = data.T[j] != '0' # has siblings or spouse on board
        data[i_nonzero,j] = 1 # map to binary: has or has not

    # parch
    j = 7
    if preprocessing[j] == 'delete':
        _delete.append(j)
    elif preprocessing[j] == 'binary':
        i_nonzero = data.T[j] != '0' # has parents or children on board
        data[i_nonzero,j] = 1 # map to binary: has or has not

    # ticket
    j = 8
    _delete.append(j)

    # fare
    # TODO: log?
    j = 9
    fare = mask_missing_data(data.T[j])
    # pclass(j=1) = 3rd class as proxy for missing fare
    class3 = data.T[1] == '3'
    data[fare.mask,j] = imputer(fare[class3], preprocessing[j])

    # cabin
    j = 10
    _delete.append(j)

    # embarked
    # embarked.mask.nonzero()[0].size = 2 (2 missing values)
    j = 11
    if preprocessing[j] == 'delete':
        _delete.append(j)
    else:
        for label, city in enumerate(CITIES):
            i = data.T[j] == city
            data[i,j] = label
        embarked = mask_missing_data(data.T[j])
        data[embarked.mask,j] = imputer(embarked, preprocessing[j])


    # delete features
    features = delete(FEATURES, _delete)
    data = delete(data, _delete, 1)

    # add features
    if unique_titles:
        features = append(features, unique_titles)
        data = append(data, _append, 1)

    return data, features

def load_and_parse_data(
        fname, preprocessing=PREPROCESSING, normalize=NORMALIZE,
        min_n=MIN_N_TITLES, suffix='', _save=False):

    # load
    data = read_data(fname)

    # test data is missing the survived column (0): append zeros
    if fname == 'test':
        data = append(zeros((len(data),1)), data, 1)

    # parse
    data, features = parse_data(data, preprocessing, min_n)

    data = data.astype(float)
    if any(isnan(data)):
        data = ma.masked_invalid(data)

    # normalize scalling
    if normalize == 'all':
        # exclude 1st column (survived)
        data = featureNormalize(data, range(1,len(data.T)))
    elif normalize:
        data = featureNormalize(data, normalize)

    # save
    if _save:
        ndim = data.shape[1]-1
        matlab_name = '_'.join(('titanic', fname))
        if not suffix:
            suffix = '_normalized'.join((preprocessing[5], normalize))
        fname = '_'.join((matlab_name, 'x%d'%ndim, suffix))
        scipy.io.savemat(save_dir + fname + '.mat', mdict={matlab_name: data})
        fname = '-'.join(('features', fname))
        #scipy.io.savemat(save_dir + fname + '.mat', mdict={'features': features})
        savetxt(save_dir + fname + '.csv', features, '%s')

    return data, features

def load_default_data(fname, featurize={3:ALL_TITLES0},
                      imputer={5:'NaN', 6:None, 7:None, 9:'NaN', 11:'NaN'}):
    preprocessing = dict(featurize.items() + imputer.items())
    normalize = []
    return load_and_parse_data(fname, preprocessing, normalize, MIN_N_TITLES)

def read_data(fname, _dir = ''):
    return loadtxt(_dir + fname + '.csv', str, delimiter=",", skiprows=1)

def read_prediction(fname, _dir = ''):
    return loadtxt(_dir + fname + '.csv', int, delimiter=",", usecols=(0,))

def write_prediction(fname, data, _dir = ''):
    return savetxt(_dir + fname + '.csv', data, '%d')

def compare_test_data(u, v):
    return mean(u == v)

def bagging(fnames, _dir='', _save=True):
    data = array([read_prediction(fname, _dir) for fname in fnames])
    majority = data.mean(axis=0) >= .5
    if _save:
        models = [x[0] for x in char.split(fnames,'.')]
        fname = '.'.join(models)
        write_prediction(fname, majority, _dir)
    return majority.astype(int)

def svm_bagging(n, _dir='', _save=True):
    fname = 'svm'
    fnames = [fname + '%d'%(i+1) for i in range(n)]
    majority = bagging(fnames, _dir)
    if _save:
        fname = 'svm_bagging%d'%n
        write_prediction(fname, majority, _dir)
    return majority

def random_forest(train_data, test_data, nTrees=100, _save=False):

    # Create the random forest object which will include all the parameters
    # for the fit
    Forest = RandomForestClassifier(n_estimators=nTrees)

    # Fit the training data to the training output and create the decision
    # trees
    X = train_data[:,1:]
    y = train_data[:,0]
    Forest = Forest.fit(X, y)

    # Take the same decision trees and run on the test data
    X = test_data[:,1:]
    y = test_data[:,0]
    predictions = Forest.predict(X)
    error = 1-compare_test_data(predictions, y)

    if _save:
        fname = 'random_forest.%d.x%d.csv' %(nTrees, X.shape[1])
        savetxt(save_dir + fname, predictions, '%d')

    return predictions, error

def cvpartition(N, K):
    bins = linspace(0,N,K+1).astype(int)
    test = [arange(bins[i], bins[i+1]) for i in range(K)]
    training = [delete(arange(N), i) for i in test]
    return training, test

def cv_randomforest(data, K, nTrees):
    training, test = cvpartition(len(data), K)
    return mean([random_forest(data[trIdx], data[teIdx], nTrees)[1]
                 for trIdx, teIdx in zip(training, test)])

def cv_ols(y, x, k=10):
    kfold = cross_validation.KFold(len(y), k)
    rmse = []
    for trIdx, teIdx in kfold:
        result = pd.ols(y=y[trIdx], x=x.loc[trIdx])
        predictions = result.predict(result.beta, x.loc[teIdx])
        error = sqrt(mean_squared_error(y[teIdx], predictions))
        rmse.append(error)
    return rmse.mean()

def new():
    # imput missing values in fare (9) and embarked (11)
    imputer = {5:'NaN', 6:None, 7:None, 9:'median', 11:'median'}
    # load and parse train set
    train, features = load_default_data('train', imputer=imputer)
    # load and parse test set
    test, features = load_default_data('test', imputer=imputer)
    # join them together and remove first column (survived)
    data = ma.masked_invalid(append(train[:,1:], test[:,1:], axis=0))
    # make dict
    d = dict(zip(features[1:], data.T))
    # make data frame
    df = pd.DataFrame.from_dict(d)
    # preprocessing: age, embarked, fare

#mapper = DataFrameMapper([
#    ('embarked', sklearn.preprocessing.LabelBinarizer())
#    ,('embarked', sklearn.preprocessing.Imputer())
#    ,('pclass', sklearn.preprocessing.LabelBinarizer())
#    ,('fare', sklearn.preprocessing.StandardScaler())
#])
