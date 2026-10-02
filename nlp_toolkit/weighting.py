#created during lab 3
import math
import numpy as np

def getcount(token, tokens): #compute_tf helper function
    count = 0
    for t in tokens:
        if t == token:
            count += 1
    return count

def compute_tf(tokens):
    tf = {}
    for token in tokens:
        tf[token] = getcount(token, tokens) / len(tokens)
    return tf

def get_document_counts(tokenized_documents):
    document_counts = {}
    for document in tokenized_documents:
        for word in set(document):
            if word not in document_counts:
                document_counts[word] = 1
            else:
                document_counts[word] += 1
    return document_counts

def compute_idf(tokenized_documents):
    N = len(tokenized_documents)
    idf = {}
    document_counts = get_document_counts(tokenized_documents)
    for document in tokenized_documents:
        for word in document:
            idf[word] = math.log((N/document_counts[word]))
    return idf

def tfidf_document(tokens, word_to_index, idf):
    tf = compute_tf(tokens)
    tfidf = [0] * len(word_to_index)
    for token in tokens:
        index = word_to_index[token]
        tfidf[index] = tf[token] * idf[token]
    return tfidf

def tfidf_corpus(tokenized_documents, word_to_index, idf):
    tfidf_documents = []
    for document in tokenized_documents:
        tfidf_documents.append(tfidf_document(document, word_to_index, idf))
    return np.array(tfidf_documents)


    