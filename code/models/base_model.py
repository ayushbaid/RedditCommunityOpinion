import abc

from typing import List


class BaseModel:
    def __init__(self):
        pass

    @abc.abstractclassmethod
    def train(self, train_sentence: List[str]):
        pass

    @abc.abstractclassmethod
    def complete(self, phrase: str):
        pass
