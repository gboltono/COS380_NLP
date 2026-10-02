#created during lab 3

import numpy as np

def cosine_similarity(vec_a, vec_b):
    numer = np.dot(np.array(vec_a), np.array(vec_b))
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if numer == 0 or denom == 0:
        return 0
    return numer / denom

def find_similar_documents(query_index, matrix, top_k=3):
    rank = []
    for doc_index in range(len(matrix)):
        if doc_index != query_index:
            a_rank = (doc_index, cosine_similarity(matrix[query_index], matrix[doc_index]))
            rank.append(a_rank)
    rank = sorted(rank, key=lambda x: x[1], reverse=True)
    return rank[:top_k]