import numpy as np
import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV

from collections import Counter
import spacy

# Toggles
figures = True

"""Import and examine dataset"""
dataset = pd.read_csv('mbpi_dataset.csv')

# Remove ||| from posts and replace them with a period. Add markers for type
dataset['posts'] = dataset['posts'].apply(lambda x: x.replace("|||"," "))

dataset['SN'] = dataset['type'].apply(lambda x: x[1])
dataset['TF'] = dataset['type'].apply(lambda x: x[2])
dataset['JP'] = dataset['type'].apply(lambda x: x[3])

# Create some helpful variables
cols = ['EI', 'SN', 'TF', 'JP']
dicts = {'E': 'Extrovert', 'I': 'Introvert', 'S': 'Sensing', 'N': 'Intuition', 'T': 'Thinking', 'F': 'Feeling',
        'J': 'Judging', 'P': 'Perceiving'}

# Create numerical metrics to quantify aspects of speech
dataset['wordspercom'] = dataset['posts'].apply(lambda x: (len(x.split())+49)/50)
dataset['ytlinks'] = dataset['posts'].apply(lambda x: (x.count('www.youtube')))
dataset['links'] = dataset['posts'].apply(lambda x: (x.count('http')))
dataset['music'] = dataset['posts'].apply(lambda x: (x.count('music')))
dataset['question'] = dataset['posts'].apply(lambda x: (x.count('?')))
dataset['ellipsis'] = dataset['posts'].apply(lambda x: (x.count('...')))
dataset['comma'] = dataset['posts'].apply(lambda x: (x.count(',')))
dataset['period'] = dataset['posts'].apply(lambda x: (x.count('.')))
dataset['semicolon'] = dataset['posts'].apply(lambda x: (x.count(';')))
dataset['right'] = dataset['posts'].apply(lambda x: (x.count('right')))
dataset['wrong'] = dataset['posts'].apply(lambda x: (x.count('wrong')))
dataset['ego'] = dataset['posts'].apply(lambda x: (x.count('I')))

# Examine distribution of personality types
prct_real = np.array([0.044,0.015,0.033,0.021,0.032,0.081,0.054,0.088,0.018,0.116,0.025,.138,0.043,0.085,0.123,0.087])
type_real = ['INFP','INFJ','INTP','INTJ','ENTP','ENFP','ISTP','ISFP','ENTJ','ISTJ','ENFJ','ISFJ','ESTP','ESFP','ESFJ','ESTJ']
real_dist = pd.DataFrame(prct_real.reshape(1, -1), index=['Percentage'], columns=type_real)
fig, axs = plt.subplots(2,1,figsize=(8,8))
sns.countplot(x='type', data=dataset,palette='colorblind', order=dataset['type'].value_counts().index, ax=axs[0])
sns.barplot(data=real_dist, palette='colorblind',ax=axs[1])
axs[0].set(xlabel='Myers Briggs Personality Type', ylabel='Dataset Count')
axs[1].set(xlabel='Myers Briggs Personality Type', ylabel='Population Percent Distribution')
fig.suptitle('Dataset and Population MBTI Distribution')
plt.tight_layout()

# Preference Comparison
fig, axs = plt.subplots(2,2,figsize=(8,8))
for i in range(2):
    for j in range(2):
        axs[i,j].bar(height=[len(dataset[dataset['type'].str.contains(cols[(2*i)+j][0])]),
                             len(dataset[dataset['type'].str.contains(cols[(2*i)+j][1])])],
                     x=(dicts[cols[(2 * i) + j][0]], dicts[cols[(2 * i) + j][1]]))
fig.suptitle('Preference Distribution')

def violinplot(df, attribute):
    plt.figure(figsize=(15,10))
    sns.violinplot(x='type', y=attribute, data=df, inner=None, color='lightgray')
    sns.stripplot(x='type', y=attribute, data=df, size=4, jitter=True)
    plt.xlabel('Myers Briggs Personality Type')
    plt.title('{0} by Personality Type'.format(attribute))

def type_comparison(df, attribute):
    fig, axs = plt.subplots(4,2, figsize=(10,10))
    cols = ['EI', 'SN', 'TF', 'JP']
    dict = {'E':'Extrovert', 'I':'Introvert', 'S':'Sensing','N':'Intuition', 'T':'Thinking','F':'Feeling', 'J':'Judging','P':'Percieving'}
    for i in range(4):
        for j in range(2):
            sns.histplot(df[df['type'].str.contains(cols[i][j])], x=attribute,  ax=axs[i,j], bins = 15, palette="crest")
            axs[i,j].set_title(dict[cols[i][j]])
    fig.suptitle('{0} by Indicator'.format(attribute))
    plt.tight_layout()

if figures:
    violinplot(dataset, 'wordspercom')
    type_comparison(dataset, 'ego')
    type_comparison(dataset, 'right')
    type_comparison(dataset, 'wrong')
    plt.show()



print('hi')


# Extract and prepare data for model training
X = dataset.drop(labels=['type','posts','EI','SN','TF','JP'], axis=1).values

# Run single and multi-lable classifiers on all preferences to determine the best model to optimize
cols = ['EI', 'SN', 'TF', 'JP', 'type']
for i in range(len(cols)):
    # y = dataset[cols[i]].values
    y = dataset[cols[i]].values

    scalar = StandardScaler().fit(X)
    X = scalar.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.2)

    # Attempt to fit a KNN, RandomForest, LogisticRegression, and SGD model
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    knn_score = knn.score(X_test, y_test)

    sgd = SGDClassifier()
    sgd.fit(X_train, y_train)
    sgd_score = sgd.score(X_test, y_test)

    logreg = LogisticRegression(max_iter=len(X_train))
    logreg.fit(X_train, y_train)
    logreg_score = logreg.score(X_test, y_test)

    randforest = RandomForestClassifier()
    randforest.fit(X_train, y_train)
    randforest_score = randforest.score(X_test, y_test)

    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    xgb_score = xgb.score(X_test, y_test)

    if i == 0:
        lst = np.array([round(xgb_score, 2), round(randforest_score, 2), round(knn_score, 2), round(logreg_score, 2), round(sgd_score, 2)])
        model_output = pd.DataFrame(lst.reshape(1, -1), index=[cols[i]], columns=['XGB', 'RandForrest', 'KNN', 'LogReg', 'SGD'])
    else:
        lst = np.array([round(xgb_score, 2), round(randforest_score, 2), round(knn_score, 2), round(logreg_score, 2),round(sgd_score, 2)])
        df = pd.DataFrame(lst.reshape(1, -1), index=[cols[i]], columns=['XGB', 'RandForrest', 'KNN', 'LogReg', 'SGD'])
        model_output = pd.concat([model_output, df])

# Compare all models
for i, col_name in enumerate(model_output.columns):
    val = 1
    for j in range(len(model_output)-1):
        val = val * model_output[col_name][j]
    if i == 0:
        lst = np.array([val])
    else:
        lst = np.hstack((lst, val))
df = pd.DataFrame(lst.reshape(1, -1), index=['Average'], columns=['XGB', 'RandForrest', 'KNN', 'LogReg', 'SGD'])
model_output = pd.concat([model_output, df])
print(model_output)

# Random Forrest Hyperparmeters
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

# Create a grid with all of the possible hyperparameters
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
rdmfrt = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rdmfrt, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Tune and fit different parameters for each indicator and for multiclassification
out = np.array(['Score', 'Classification'])
for i in range(len(cols)):
    y = dataset[cols[i]].values
    scalar = StandardScaler().fit(X)
    X = scalar.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2)
    rf_random.fit(X_train, y_train)
    parameters = rf_random.best_params_
    neigh = RandomForestClassifier(max_features=parameters["max_features"], n_estimators=parameters["n_estimators"],
                             min_samples_split=parameters["min_samples_split"], min_samples_leaf=parameters["min_samples_leaf"],
                                   bootstrap=parameters['bootstrap'])
    neigh.fit(X_train, y_train)
    score = neigh.score(X_test, y_test)
    out = np.vstack((out, [score, cols[i]]))
print(model_output)
print(out)