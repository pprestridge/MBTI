import spacy
import pandas as pd
from spacy.tokens import DocBin
import pickle
from collections import Counter

# Import Dataset
dataset = pd.read_csv('mbpi_dataset.csv')
dataset['posts'] = dataset['posts'].apply(lambda x: x.replace("|||",". "))

# Spacy nlp processing
nlp = spacy.load("en_core_web_lg")
dataset['doc'] = dataset['posts'].apply(lambda x: nlp(x))

# Compile all docs in a DocBin and serialize
doc_bin = DocBin()
dataset['doc'].apply(lambda x: doc_bin.add(x))
doc_bin_bytes = doc_bin.to_bytes()

# Save the serialized file with pickle
pickle_out = open("nlpDocs.pickle","wb")
pickle.dump(doc_bin_bytes, pickle_out)
pickle_out.close()

# Report the most common words used based on Part of Speech
def most_common(doc, num, type):
    if type == 'word':
        words = [token.lemma_ for token in doc if
                 token.is_stop != True and token.is_punct != True and token.is_space != True]
    if type == 'noun':
        words = [token.lemma_ for token in doc if token.is_stop != True and token.is_punct != True and token.pos_ == "NOUN"]
    if type == 'verb':
        words = [token.lemma_ for token in doc if token.is_stop != True and token.is_punct != True and token.pos_ == "VERB"]
    if type == 'adjective':
        words = [token.lemma_ for token in doc if token.is_stop != True and token.is_punct != True and token.pos_ == "ADJ"]
    return words

# Update the df with the 5 most common nouns, verbs, and adjectives of each user
dataset['words'] = dataset['doc'].apply(lambda x: most_common(x, 5, 'word'))
dataset['nouns'] = dataset['doc'].apply(lambda x: most_common(x, 5, 'noun'))
dataset['verbs'] = dataset['doc'].apply(lambda x: most_common(x, 5, 'verb'))
dataset['adjectives'] = dataset['doc'].apply(lambda x: most_common(x, 5, 'adjective'))
dataset.drop(labels = ['doc'],axis=1, inplace = True)
dataset.to_csv('NLP_data.csv')