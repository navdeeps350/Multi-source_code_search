import pandas as pd
import numpy as np
import re
from statistics import mean 

import gensim

from gensim.similarities import SparseMatrixSimilarity, MatrixSimilarity
from collections import defaultdict
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LsiModel
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE



file = input("Please enter the file name: ")
ground_truth = input("Please enter the ground truth file name: ")

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
lsi = LsiModel(corpus_tfidf, id2word=dictionary, num_topics=300)
corpus_lsi = lsi[corpus_tfidf]

index_tfidf = SparseMatrixSimilarity(tfidf[corpus_bow], num_features=len(dictionary))
index_lsi = MatrixSimilarity(corpus_lsi)
index_bow = SparseMatrixSimilarity(corpus_bow, num_features=len(dictionary))


df_3 = pd.read_csv(file)  
df_3.replace(np.nan,"", regex=True, inplace=True)



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


filename = 'doc2vec.model'
model_doc2vec = gensim.models.doc2vec.Doc2Vec.load(filename)



ground_truth_list = []
with open(ground_truth) as f:
    [ground_truth_list.append(line.strip()) for line in f.readlines()]

ground_truth_list = list(filter(None, ground_truth_list))

query_list = ground_truth_list[::3]

name_list = ground_truth_list[1::3]

file_list = ground_truth_list[2::3]


bow_tp = 0
tfidf_tp = 0
lsi_tp = 0
doc2vec_tp = 0

prec_bow = []
prec_tfidf = []
prec_lsi = []
prec_doc2vec = []

for query, name, file in zip(query_list, name_list, file_list):
    tfidf_name = []
    tfidf_file = []
    bow_name = []
    bow_file = []
    lsi_name = []
    lsi_file = []
    doc2vec_name = []
    doc2vec_file = []
    query_bow = dictionary.doc2bow(query.lower().split())
    sims_tfidf = index_tfidf[tfidf[query_bow]]
    for idx, score in sorted(enumerate(sims_tfidf), key=lambda x: x[1], reverse=True)[:5]:
        tfidf_name.append(df_3['name'].iloc[idx])
        tfidf_file.append(df_3['file'].iloc[idx])

    sims_lsi = index_lsi[lsi[tfidf[query_bow]]]
    for idx, score in sorted(enumerate(sims_lsi), key=lambda x: x[1], reverse=True)[:5]:
        lsi_name.append(df_3['name'].iloc[idx])
        lsi_file.append(df_3['file'].iloc[idx])
        
    sims_bow = index_bow[query_bow]
    for idx, score in sorted(enumerate(sims_bow), key=lambda x: x[1], reverse=True)[:5]:
        bow_name.append(df_3['name'].iloc[idx])
        bow_file.append(df_3['file'].iloc[idx])

    test_corpus = list(read_corpus([query.lower()], tokens_only=True))
    inferred_vector = model_doc2vec.infer_vector(test_corpus[0])
    sims_doc2vec = model_doc2vec.dv.most_similar([inferred_vector], topn=len(model_doc2vec.dv))

    for idx in list(map(lambda x: x[0], sims_doc2vec[:5])):
        doc2vec_name.append(df_3['name'].iloc[idx])
        doc2vec_file.append(df_3['file'].iloc[idx])


    if name in tfidf_name:
        tfidf_tp += 1
        POS = tfidf_name.index(name) + 1
        prec_tfidf.append(1/POS)
    else:
        POS = 0
        prec_tfidf.append(0)

    if name in lsi_name:
        lsi_tp += 1
        POS = lsi_name.index(name) + 1
        prec_lsi.append(1/POS)
    else:
        POS = 0
        prec_lsi.append(0)

    if name in bow_name:
        bow_tp += 1
        POS = bow_name.index(name) + 1
        prec_bow.append(1/POS)
    else:
        POS = 0
        prec_bow.append(0)

    if name in doc2vec_name:
        doc2vec_tp += 1
        POS = doc2vec_name.index(name) + 1
        prec_doc2vec.append(1/POS)
    else:
        POS = 0
        prec_doc2vec.append(0)
    

print('TF-IDF')
print('Average precision: ', mean(prec_tfidf))
recall_tfidf = tfidf_tp/len(query_list)
print('Recall: ', recall_tfidf)

print('LSI')
print('Average precision: ', mean(prec_lsi))
recall_lsi = lsi_tp/len(query_list)
print('Recall: ', recall_lsi)

print('Bag of Words')
print('Average precision: ', mean(prec_bow))
recall_bow = bow_tp/len(query_list)
print('Recall: ', recall_bow)

print('Doc2Vec')
print('Average precision: ', mean(prec_doc2vec))
recall_doc2vec = doc2vec_tp/len(query_list)
print('Recall: ', recall_doc2vec)


list_corpus_lsi = list(corpus_lsi)
col_c = np.repeat(np.linspace(0, 9, 10), 6)


inferred_vectors_lsi = []
inferred_vectors_doc2vec = []

for query, name, file in zip(query_list, name_list, file_list):
    query_bow = dictionary.doc2bow(query.lower().split())
    inferred_vectors_lsi.append(lsi[tfidf[query_bow]])
    sims_lsi = index_lsi[lsi[tfidf[query_bow]]]
    sims_lsi = abs(sims_lsi)
    for idx, score in sorted(enumerate(sims_lsi), key=lambda x: x[1], reverse=True)[:5]:
        inferred_vectors_lsi.append(list_corpus_lsi[idx])
    test_corpus = list(read_corpus([query.lower()], tokens_only=True))
    query_vector_doc2vec = model_doc2vec.infer_vector(test_corpus[0])
    inferred_vectors_doc2vec.append(query_vector_doc2vec)
    sims_doc2vec = model_doc2vec.dv.most_similar([query_vector_doc2vec], topn=len(model_doc2vec.dv))
    for idx in list(map(lambda x: x[0], sims_doc2vec[:5])):
        inferred_vectors_doc2vec.append(model_doc2vec.infer_vector(train_corpus[idx][0]))


vect = np.array(inferred_vectors_lsi)[:, :, 1].reshape(np.array(inferred_vectors_lsi).shape[0], np.array(inferred_vectors_lsi).shape[1])
tsne = TSNE(n_components=2, verbose=1, perplexity=2, n_iter=3000)
tsne_red = tsne.fit_transform(vect)

scatter = plt.scatter(tsne_red[:, 0], tsne_red[:, 1], c = col_c )
handles, _ = scatter.legend_elements(prop='colors')
labels = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10']
plt.legend(handles, labels)

plt.show()



tsne = TSNE(n_components=2, verbose=1, perplexity=2, n_iter=3000)
tsne_red_doc = tsne.fit_transform(np.array(inferred_vectors_doc2vec))

scatter = plt.scatter(tsne_red_doc[:, 0], tsne_red_doc[:, 1], c = col_c )
handles, _ = scatter.legend_elements(prop='colors')
labels = ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10']
plt.legend(handles, labels)

plt.show()
