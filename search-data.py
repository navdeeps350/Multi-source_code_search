import pandas as pd
import numpy as np
import re

import gensim

from gensim.similarities import SparseMatrixSimilarity, MatrixSimilarity
from collections import defaultdict
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LsiModel
import matplotlib.pyplot as plt


file = input("Please enter the file name: ")
query = input("Please enter the query: ")


df = pd.read_csv(file)  

def filtered_words(name, stop_words):
    filtered_words = [] 
    # Split by camel case
    words = re.findall(r'[a-zA-Z][a-z]*|[A-Z][a-z]*', name)
    
    # Split by underscore
    words = [word.split('_') for word in words]
    words = sum(words, [])

    # Iterate over the list of words 
    for word in words:
    # If the word is not in the stop word list, add it to the filtered list 
        if word not in stop_words:
            filtered_words.append(word) 
         
    
    return filtered_words


def filtering(name_list, stop_words):
    filtered_words = [] 

    # Iterate over the list of words 
    for word in name_list: 
    # If the word is not in the stop word list, add it to the filtered list 
        if word not in stop_words: 
            filtered_words.append(word)
    return filtered_words

stop_words = ['test', 'tests', 'main', 'this', 'is', 'in', 'to'] 

for i, name in enumerate(df['name']):
    df['name'][i] = filtered_words(name, stop_words)


df.replace(np.nan,"", regex=True, inplace=True)


for i, comment in enumerate(df['comment']):
    df['comment'][i] = filtering(comment.split(), stop_words)


df_1 = df[['name', 'comment']].copy()

df_2  = df_1['name'] + df_1['comment']

text = list(df_2)

def all_lower(my_list):
    return [[x.lower() for x in list] for list in my_list]

text_lower = all_lower(text)

corpus_text = [[' '.join(my_list)] for my_list in text_lower]


frequency = defaultdict(int)
for text in text_lower:
    for token in text:
        frequency[token] += 1


dictionary = Dictionary(text_lower)

corpus_bow = [dictionary.doc2bow(text) for text in text_lower]

tfidf = TfidfModel(corpus_bow)
corpus_tfidf = tfidf[corpus_bow]
index_tfidf = SparseMatrixSimilarity(corpus_tfidf, num_features=len(dictionary))

lsi = LsiModel(corpus_tfidf, id2word=dictionary, num_topics=300)
corpus_lsi = lsi[corpus_tfidf]
index_lsi = MatrixSimilarity(corpus_lsi)

index_bow = SparseMatrixSimilarity(corpus_bow, num_features=len(dictionary))


# index_tfidf = SparseMatrixSimilarity(tfidf[corpus_bow], num_features=len(dictionary))

query_bow = dictionary.doc2bow(query.lower().split())


sims_tfidf = index_tfidf[tfidf[query_bow]]
sims_lsi = index_lsi[lsi[tfidf[query_bow]]]
sims_lsi = abs(sims_lsi)

sims_bow = index_bow[query_bow]


df_3 = pd.read_csv(file)  
df_3.replace(np.nan,"", regex=True, inplace=True)

print("Sorted by tf-idf scores")
for idx, score in sorted(enumerate(sims_tfidf), key=lambda x: x[1], reverse=True)[:5]:
    print(idx, score)
    print(df_3.iloc[idx])

# vec_bow = dictionary.doc2bow(query.lower().split())
# vec_lsi = lsi[tfidf[query_bow]]

# index_lsi = MatrixSimilarity(corpus_lsi)
# sims_lsi = index_lsi[lsi[tfidf[query_bow]]]
# sims_lsi = abs(sims_lsi)


print("Sorted by lsi scores")
for idx, score in sorted(enumerate(sims_lsi), key=lambda x: x[1], reverse=True)[:5]:
    print(idx, score)
    print(df_3.iloc[idx])

# index_bow = SparseMatrixSimilarity(corpus_bow, num_features=len(dictionary))
# sims_bow = index_bow[query_bow]


print("Sorted by bow scores")
for idx, score in sorted(enumerate(sims_bow), key=lambda x: x[1], reverse=True)[:5]:
    print(idx, score)
    print(df_3.iloc[idx])


def read_corpus(text_corpus, tokens_only=False):
    for i, line in enumerate(text_corpus):
        if tokens_only:
            tokens = gensim.utils.simple_preprocess(line)
            yield tokens
        else:
            # For training data, add tags
            tokens = gensim.utils.simple_preprocess(line[0])
            yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

train_corpus = list(read_corpus(corpus_text))



test_corpus = list(read_corpus([query], tokens_only=True))

model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
model.build_vocab(train_corpus)
model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

inferred_vector = model.infer_vector(test_corpus[0])

sims_doc2vec = model.dv.most_similar([inferred_vector], topn=len(model.dv))


print("Sorted by doc2vec scores")
for i in list(map(lambda x: x[0], sims_doc2vec[:5])):
    print(df_3.iloc[i])

filename = 'doc2vec.model'
model.save(filename)
