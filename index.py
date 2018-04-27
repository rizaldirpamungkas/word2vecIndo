from __future__ import absolute_import, division, print_function
import logging
import re
import nltk
import multiprocessing
import gensim.models.word2vec as w2v
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nltk.corpus import brown

def getMatrix(matrixName):
    return pd.read_csv(matrixName)

def getModel(modelName):
    return w2v.Word2Vec.load(modelName)

# Proses pembersihan corpus

def sentenceToWordlist(raw):
    clean = re.sub("[^a-zA-Z]"," ", raw)
    words = clean.split()
    return words

# Method untuk membangun model latihan w2v, saat ini corpus yang dipakai adalah brown, dengan ukuran
# vocab 21 ribu kata, proses ini cukup memakan waktu bisa mencapai setengah jam

def build_model():
    handle = open("doksatu.txt",'r')
    raw_corpus = handle.read().casefold()
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    raw_sentences = tokenizer.tokenize(raw_corpus)


    sentences = []
    for raw_sentence in raw_sentences:
        if len(raw_sentence) > 0:
            sentences.append(sentenceToWordlist(raw_sentence))
    
    # print(sentences)

    num_features = 300
    min_word_count = 3
    num_workers = multiprocessing.cpu_count()
    context_size = 7
    downsampling = 1e-3
    seed = 1

    wikipediaID = w2v.Word2Vec(
        sg=True,
        seed=seed,
        workers=num_workers,
        size=num_features,
        min_count=min_word_count,
        window=context_size,
        sample=downsampling
    )

    wikipediaID.build_vocab(sentences)

    wikipediaID.train(sentences,total_examples=wikipediaID.corpus_count, epochs=wikipediaID.epochs)
    wikipediaID.save("wikipediaID.w2v")

def reduceDimensionality(nameModel):  
    wikipediaID = getModel(nameModel)
    tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
    all_word_vectors_matrix = wikipediaID.wv.syn0
    all_word_vectors_matrix_2d = tsne.fit_transform(all_word_vectors_matrix)

    points = pd.DataFrame(
        [
            (word, coords[0], coords[1])
            for word, coords in [
                (word, all_word_vectors_matrix_2d[wikipediaID.wv.vocab[word].index])
                for word in wikipediaID.wv.vocab
            ]
        ],
        columns=["word", "x", "y"]
    )
    points.to_csv('model.csv')

def showBigPicture():
    points = getMatrix('model.csv')
    fig, ax = plt.subplots()
    ax.scatter(points['x'], points['y'])

    for i, txt in enumerate(points['word']):
        ax.annotate(txt, (points['x'][i], points['y'][i]))
    
    points.plot.scatter("x", "y", s=10, figsize=(30, 18))
    plt.show()


# build_model()
# reduceDimensionality('wikipediaID.w2v')


# print(len(getModel('wikipediaID.w2v').wv.vocab))
showBigPicture()
# print(getModel('wikipediaID.w2v').most_similar('liverpool'))
# print(getModel('wikipediaID.w2v').similarity('liverpool','asam'))