from gensim.models import KeyedVectors

import torch
import torch.nn as nn

from models.base_model import BaseModel


class Word2VecModel(nn.Module):
    def __init__(self, context_size: int = 2):
        super().__init__()

        vocab_size = 3000000
        embedding_dim = 300

        self.model = nn.Sequential(
            nn.Embedding.from_pretrained(torch.load(
                '../models/google_news_word2vec.pt')),
            nn.Linear(context_size*embedding_dim, 128),
            nn.Linear(128, vocab_size)
        )

    def forward(self, inputs):
        pass
