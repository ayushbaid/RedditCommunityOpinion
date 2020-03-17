import numpy as np


def restrict_w2v(w2v, restricted_word_set):
    new_vectors = []
    new_vocab = {}
    new_index2entity = []
    #new_vectors_norm = []

    for i in range(len(w2v.wv.vocab)):
        # import pdb
        # pdb.set_trace()
        word = w2v.wv.index2entity[i]
        vec = w2v.vectors[i]
        vocab = w2v.wv.vocab[word]
        #vec_norm = w2v.vectors_norm[i]
        if word in restricted_word_set:
            vocab.index = len(new_index2entity)
            new_index2entity.append(word)
            new_vocab[word] = vocab
            new_vectors.append(vec)
            # new_vectors_norm.append(vec_norm)

    w2v.vocab = new_vocab
    w2v.vectors = np.array(new_vectors)
    w2v.index2entity = np.array(new_index2entity)
    w2v.index2word = np.array(new_index2entity)
    #w2v.vectors_norm = np.array(new_vectors_norm)
