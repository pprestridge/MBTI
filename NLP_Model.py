import numpy as np
import pandas as pd
import os
import pickle
from spacy.tokens import DocBin
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import spacy
from spacy.vectors import Vectors


# Import Dataset
dataset = pd.read_csv('NLP_Data.csv')

# Clean saved dataset by recreating word lists (Saving issues)
unclear_cols = ['words', 'nouns', 'adjectives', 'verbs']
for i, var_name in enumerate(unclear_cols):
    dataset[var_name] = dataset[var_name].apply(lambda x: (x.replace("'", '').replace("[", '').split(",")))

# Import NLP-Processed Data
# nlp = spacy.load("en_core_web_lg")
# pickle_in = open("nlpDocs_small.pickle","rb")
# doc_bin_bytes = pickle.load(pickle_in)
# doc_bin = DocBin().from_bytes(doc_bin_bytes)
# docs = list(doc_bin.get_docs(nlp.vocab))
# dataset = dataset[:len(docs)]
# dataset['doc'] = docs

# Useful Variables
cols = ['EI', 'SN', 'TF', 'JP']
all_types = ['Extrovert',  'Introvert', 'Sensing',  'Intuition',  'Thinking',  'Feeling','Judging',  'Perceiving']
dicts = {'E': 'Extrovert', 'I': 'Introvert', 'S': 'Sensing', 'N': 'Intuition', 'T': 'Thinking', 'F': 'Feeling',
        'J': 'Judging', 'P': 'Perceiving'}

# Compare word usage across MBPI Types
def word_compare(df, col, filt):
    fig, axs = plt.subplots(4,2, figsize=(10,10))
    for i in range(4):
        set1 = df[dataset['type'].str.contains(cols[i][0])]
        list1 = list()
        set1[col].apply(lambda x: list1.extend(x))
        set2 = df[dataset['type'].str.contains(cols[i][1])]
        list2 = list()
        set2[col].apply(lambda x: list2.extend(x))
        list_comb = [list1,list2]
        if filt:
            filt_dfs, repeats = word_filter(list_comb)
        for j in range(2):
            freq = Counter(list_comb[j])
            freq_df = pd.DataFrame.from_dict(freq, orient='index')
            freq_df = freq_df.sort_values(by=0, ascending=False)
            freq_df = freq_df[0:10]
            if filt:
                freq_df = filt_dfs[j]
                if len(freq_df) >= 10:
                    freq_df = freq_df[0:10]
            freq_df = freq_df.iloc[::-1]
            freq_df.plot.barh(ax=axs[i, j])
            axs[i, j].set_title(dicts[cols[i][j]])
            axs[i, j].get_legend().remove()
            file_title = str(cols[i][j] +'_target_' + col + '.csv')
            file_out = pd.DataFrame(freq_df.index)
            path = "C:\\Users\\Patrick\\PycharmProjects\\PLPProjects\\Storage"
            path = os.path.join(path, file_title)
            file_out.to_csv(path)
        file_title = str(cols[i] +'_repeats_' + col + '.csv')
        path = "C:\\Users\\Patrick\\PycharmProjects\\PLPProjects\\Storage"
        path = os.path.join(path, file_title)
        file_out = pd.DataFrame(repeats)
        file_out.to_csv(path)
    fig.suptitle('Most common {0} by Type'.format(col))
    plt.tight_layout()

# Remove common vocabulary for a better comparison
def word_filter(list_comb):
    freq1 = Counter(list_comb[0])
    freq2 = Counter(list_comb[1])
    freq_df1 = pd.DataFrame.from_dict(freq1, orient='index')
    freq_df2 = pd.DataFrame.from_dict(freq2, orient='index')
    freq_df1 = freq_df1.sort_values(by=0, ascending=False)
    freq_df2 = freq_df2.sort_values(by=0, ascending=False)
    freq_df1 = freq_df1[0:500]
    freq_df2 = freq_df2[0:500]
    repeats = list()
    for i, var_name in enumerate(freq_df1.index):
        if var_name in freq_df2.index:
            repeats.append([var_name])
    for i in range(len(repeats)):
        freq_df1.drop(repeats[i][0], inplace = True)
        freq_df2.drop(repeats[i][0], inplace = True)
    freq_dfs = [freq_df1, freq_df2]
    return freq_dfs, repeats

# Run each category through word_compare() and word_filter() to determine words to discard and words to use for
# similarity analysis
if False:
    for i, var_name in enumerate(unclear_cols):
        word_compare(dataset, var_name, filt=True)
    plt.show()

def word_retriever(word_type, indicator):
    path = "C:\\Users\\Patrick\\PycharmProjects\\PLPProjects\\Storage"
    file_title = str(indicator[0] + '_target_' + word_type + '.csv')
    file_path = os.path.join(path, file_title)
    target_cat1 = pd.read_csv(file_path)
    file_title = str(indicator[1] + '_target_' + word_type + '.csv')
    file_path = os.path.join(path, file_title)
    target_cat2 = pd.read_csv(file_path)
    file_title = str(indicator + '_repeats_' + word_type + '.csv')
    file_path = os.path.join(path, file_title)
    repeats = pd.read_csv(file_path)
    repeats
    return np.array(repeats['0']), np.array(target_cat1['0']), np.array(target_cat2['0'])
def word_remover(df_row, word_type, repeated_words, i):
    words_list = df_row[word_type][i]
    freq = Counter(words_list)
    freq_df = pd.DataFrame.from_dict(freq, orient='index')
    freq_df = freq_df.sort_values(by=0, ascending=False)
    for j in range(len(repeated_words)):
        if repeated_words[j] in freq_df.index:
            freq_df.drop(repeated_words[j], inplace = True)
    freq_df = freq_df[freq_df>=2]
    freq_df.dropna(inplace = True)
    if len(freq_df) > 10:
        freq_df = freq_df[:10]
    return np.array(freq_df.index)
def predictor(words, target1, target2, w2vec):
    col_len = 300

    for i, word in enumerate(words):
        vec = w2vec(word)
        for j in range(10):
            targ_vec = w2vec(target1[j])
            targ_vec2 = w2vec(target2[j])
            score = targ_vec.similarity(vec)
            score2 = targ_vec2.similarity(vec)
            if 'culmscore1' in locals():
                culmscore1 = np.vstack((culmscore1, score))
                culmscore2 = np.vstack((culmscore2, score))
            else:
                culmscore1 = np.array(score)
                culmscore2 = np.array(score2)

    if 'culmscore1' in locals():
        score1 = np.mean(culmscore1)
        score2 = np.mean(culmscore2)
        if score1>score2:
            return 0
        else:
            return 1
    else:
        return 1


# Prepare spacey word to vector library
w2vec = spacy.load('en_vectors_web_lg')


def NLP_Model(data_frame,iter,w2vec):
    # Loop to run analysis by each word type
    for i, word_type in enumerate(unclear_cols):
        # Loop to run analysis by indicator E vs. I
        for j, indicator in enumerate(cols):
            # Loop to run analysis by each category
            repeated_words, target_words1, target_words2 = word_retriever(word_type,indicator)
            input_words = word_remover(data_frame, word_type, repeated_words, iter)
            output_category = predictor(input_words, target_words1, target_words2, w2vec)

            out_dict = {word_type+indicator: [indicator[output_category]]}
            if i == 0 and j == 0:
                out_df = pd.DataFrame.from_dict(out_dict)
            else:
                add_df = pd.DataFrame.from_dict(out_dict)
                out_df = pd.concat([out_df, add_df], axis=1)

    print('hi')
    return out_df

for i in range(len(dataset)):
    prediction = NLP_Model(dataset, i, w2vec)
    if i == 0:
        model_output = prediction
    else:
        model_output = pd.concat([model_output, prediction], axis=0)

model_output.to_csv('NLP_model_prediction .csv')
print('hi')