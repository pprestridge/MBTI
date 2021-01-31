import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import Dataset
model_out = pd.read_csv('NLP_model_prediction.csv')
solution_set = pd.read_csv('mbpi_dataset.csv')
model_out["Unnamed: 0"] = solution_set['type']
model_out.rename(columns = {'Unnamed: 0':'type'}, inplace = True)
cols = ['EI', 'SN', 'TF', 'JP']

# Select relevant model slices
colInd = model_out.columns
EI = model_out.iloc[:,colInd.str.contains('EI')]
SN = model_out.iloc[:,colInd.str.contains('SN')]
TF = model_out.iloc[:,colInd.str.contains('TF')]
JP = model_out.iloc[:,colInd.str.contains('JP')]
words = model_out.iloc[:,colInd.str.contains('words')]
nouns = model_out.iloc[:,colInd.str.contains('nouns')]
adjectives = model_out.iloc[:,colInd.str.contains('adjectives')]
verbs = model_out.iloc[:,colInd.str.contains('verbs')]

# Function to run through each label and score the models
def model_scoring(y, out, type):

    cols = ['EI', 'SN', 'TF', 'JP']
    word_types = ['words','nouns','adjectives','verbs']
    for i in range(len(cols)):
        if cols[i] == type:
            ind = i
    correct = y.apply(lambda x: x[ind])

    for i, label in enumerate(out.columns):
        x1, x2, x3, x4 = 0, 0, 0, 0
        for j in range(len(out)):
            x1, x2, x3, x4 = compare_outs(correct[j], out[label][j], type,x1, x2, x3, x4)
        conf_matrix_plotter(x1, x2, x3, x4,word_types[i], type)

# Function to score output and return confusion matrix parameters
def compare_outs(correct, given, type,x1, x2, x3, x4):
    if correct == type[0]:
        if correct == given:
            x1 = x1+1
        else:
            x2 = x2+1

    if correct == type[1]:
        if correct == given:
            x4 = x4+1
        else:
            x3 = x3+1

    return x1, x2, x3, x4

# Calculate Accuracy and plot confusion matrix
def conf_matrix_plotter(x1, x2, x3, x4,word_type, type):
    col1 = ['Predicted: '+ type[0]]
    col2 = ['Predicted: '+ type[1]]
    row1 = ['Actual: '+ type[0]]
    row2 = ['Actual: '+ type[0]]
    df = pd.DataFrame(data=np.array([[x1, x2], [x3, x4]]), index=np.array([row1, row2]), columns=np.array([col1, col2]))
    plt.figure()
    sns.heatmap(df, annot=True, fmt='d',cmap="YlGnBu")
    plt.title('Predicting {0} vs. {1} by {2}. Accuracy: {3}%'.format(
        type[0], type[1], word_type, round((x1+x4)/(x1+x2+x3+x4)*100,2)))



model_scoring(model_out['type'], EI, 'EI')
model_scoring(model_out['type'], SN, 'SN')
model_scoring(model_out['type'], TF, 'TF')
model_scoring(model_out['type'], JP, 'JP')
plt.show()
