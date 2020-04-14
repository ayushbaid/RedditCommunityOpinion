from typing import List, Set

import torch
import numpy as np

from nltk.util import everygrams
from nltk.tokenize import wordpunct_tokenize


def get_vocab_set(inp: List[str]) -> Set[str]:
    # break the sentences to word
    vocab_set = set()

    for inp_sentence in inp:
        for word in wordpunct_tokenize(inp_sentence):
            vocab_set.add(word)

    return vocab_set


def convert_sentence_to_ngrams(inp_sentence: str, n_param=3, add_unknown=False) -> List[str]:
    '''
    Convert the input sentence to trigrams using a tokenizer
    '''
    tokenized_input = wordpunct_tokenize(inp_sentence)

    # and an unknown token behind the
    if add_unknown:
        tokenized_input = ['<UNK>'] + tokenized_input

    return everygrams(tokenized_input, min_len=n_param, max_len=n_param)


def embed_words(inp_words: List[str], word2vec_model) -> torch.FloatTensor:
    '''
    Embeds the input words and concatenates the embeddings to create a final tensor

    '''
    try:
        return torch.FloatTensor(
            np.hstack([word2vec_model[word] if word !=
                       '<UNK>' else np.zeros((300,)) for word in inp_words])
        )
    except KeyError:
        return None


def fetch_index(word: str, word2vec_model) -> int:
    vocab_target = word2vec_model.wv.vocab.get(word)
    if vocab_target is None:
        return None

    return vocab_target.index
