import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from collections import Counter
import spacy

# Toggles
figures = False

"""Import and examine dataset"""
dataset = pd.read_csv('mbpi_dataset.csv')

# Remove ||| from posts and replace them with a period. Add markers for type
dataset['posts'] = dataset['posts'].apply(lambda x: x.replace("|||",". "))
dataset['EI'] = dataset['type'].apply(lambda x: x[0])
dataset['SN'] = dataset['type'].apply(lambda x: x[1])
dataset['TF'] = dataset['type'].apply(lambda x: x[2])
dataset['JP'] = dataset['type'].apply(lambda x: x[3])

# Create some helpful variables
cols = ['EI', 'SN', 'TF', 'JP']
dict = {'E': 'Extrovert', 'I': 'Introvert', 'S': 'Sensing', 'N': 'Intuition', 'T': 'Thinking', 'F': 'Feeling',
        'J': 'Judging', 'P': 'Perceiving'}

# Create numerical metrics to quanitify aspects of speech
dataset['wordspercom'] = dataset['posts'].apply(lambda x: (len(x.split())+49)/50)
dataset['ytlinks'] = dataset['posts'].apply(lambda x: (x.count('www.youtube')))
dataset['links'] = dataset['posts'].apply(lambda x: (x.count('http')))
dataset['music'] = dataset['posts'].apply(lambda x: (x.count('music')))
dataset['question'] = dataset['posts'].apply(lambda x: (x.count('?')))
dataset['ellipsis'] = dataset['posts'].apply(lambda x: (x.count('...')))
dataset['comma'] = dataset['posts'].apply(lambda x: (x.count(',')))
dataset['period'] = dataset['posts'].apply(lambda x: (x.count('.')))
dataset['semicolon'] = dataset['posts'].apply(lambda x: (x.count(';')))

# Examine distribution of personality types
fig = plt.figure(figsize=(8,6))
ax = sns.countplot(x='type', data=dataset,palette='colorblind', order=dataset['type'].value_counts().index)
ax.set(xlabel='Myers Briggs Personality Type', ylabel='Count')

# Preference Comparison
fig, axs = plt.subplots(2,2,figsize=(8,8))
for i in range(2):
    for j in range(2):
        axs[i,j].bar(height=[len(dataset[dataset['type'].str.contains(cols[(2*i)+j][0])]),
                             len(dataset[dataset['type'].str.contains(cols[(2*i)+j][1])])],
                     x=(dict[cols[(2*i)+j][0]], dict[cols[(2*i)+j][1]]))
fig.suptitle('Preference Distribution')

def violinplot(df, attribute):
    plt.figure(figsize=(15,10))
    sns.violinplot(x='type', y=attribute, data=df, inner=None, color='lightgray')
    sns.stripplot(x='type', y=attribute, data=df, size=4, jitter=True)
    plt.title('{0} by Personality Type'.format(attribute))

def type_comparison(df, attribute):
    fig, axs = plt.subplots(4,2, figsize=(10,10))
    cols = ['EI', 'SN', 'TF', 'JP']
    dict = {'E':'Extrovert', 'I':'Introvert', 'S':'Sensing','N':'Intuition', 'T':'Thinking','F':'Feeling', 'J':'Judging','P':'Percieving'}
    for i in range(4):
        for j in range(2):
            sns.histplot(df[df['type'].str.contains(cols[i][j])], x=attribute,  ax=axs[i,j], bins = 15)
            axs[i,j].set_title(dict[cols[i][j]])
    fig.suptitle('{0} by type'.format(attribute))
    plt.tight_layout()

if figures:
    num_attributes = dataset.columns[6:]
    for i, attribute in enumerate(num_attributes):
        violinplot(dataset, attribute)

    type_comparison(dataset, 'wordspercom')
    type_comparison(dataset, 'links')
    type_comparison(dataset, 'ytlinks')
    plt.show()

# Extract and prepare data for model training
X = dataset.drop(labels=['type','posts','EI','SN','TF','JP'], axis=1).values
for i in range(4):
    y = dataset[cols[i]].values
    print('----- Prediction for {0} ----- '.format(cols[i]))

    scalar = MinMaxScaler().fit(X)
    X = scalar.transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.2)

    # Attempt to fit a KNN, RandomForest, LogisticRegression, and SGD model
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    knn_score = knn.score(X_test, y_test)
    print('KNN Score: {0}'.format(round(knn_score,2)))

    sgd = SGDClassifier()
    sgd.fit(X_train, y_train)
    sgd_score = sgd.score(X_test, y_test)
    print('SGD Score: {0}'.format(round(sgd_score,2)))

    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    logreg_score = logreg.score(X_test, y_test)
    print('Logistic Regression Score: {0}'.format(round(logreg_score,2)))

    randforest = RandomForestClassifier()
    randforest.fit(X_train, y_train)
    randforest_score = randforest.score(X_test, y_test)
    print('Random Forest Score: {0}'.format(round(randforest_score,2)))


