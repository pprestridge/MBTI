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

