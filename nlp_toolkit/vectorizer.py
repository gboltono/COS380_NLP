import numpy as np
def build_vocabulary(tokenized_documents):
    vocabulary = []
    for document in tokenized_documents:
        for token in document:
            if token not in vocabulary:
                vocabulary.append(token)
    return vocabulary

def create_word_to_index(vocabulary):
    word_to_index = {}
    for i in range(len(vocabulary)):
        word_to_index[vocabulary[i]] = i
    return word_to_index

def vectorize_document(tokens, word_to_index):
    counts = [0] * len(word_to_index)
    for token in tokens:
        for word, index in word_to_index.items():
            if token == word:
                counts[index] += 1
    return counts

def vectorize_corpus(tokenized_documents, word_to_index):
    matrix = []
    for document in tokenized_documents:
        doc = vectorize_document(document, word_to_index)
        matrix.append(doc)
    return np.array(matrix)

#print(vectorize_corpus([['cat', 'pizza', 'chair', 'dog'],['dog','pizza','pizza'],['pizza', 'chair', 'dog']], {'cat': 0, 'pizza': 1, 'chair': 2, 'dog': 3}))