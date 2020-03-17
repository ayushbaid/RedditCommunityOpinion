from typing import List

import torch
import numpy as np

from nltk.util import everygrams
from nltk.tokenize import wordpunct_tokenize


def convert_sentence_to_ngrams(inp_sentence: str, n_param=3) -> List[str]:
    '''
    Convert the input sentence to trigrams using a tokenizer
    '''
    return everygrams(wordpunct_tokenize(inp_sentence), min_len=n_param, max_len=n_param)


def embed_words(inp_words: List[str], word2vec_model) -> torch.FloatTensor:
    '''
    Embeds the input words and concatenates the embeddings to create a final tensor

    '''
    try:
        return torch.FloatTensor(
            np.hstack([word2vec_model[word] for word in inp_words])
        )
    except KeyError:
        return None


def fetch_index(word: str, word2vec_model) -> int:
    vocab_target = word2vec_model.wv.vocab.get(word)
    if vocab_target is None:
        return None

    return vocab_target.index
