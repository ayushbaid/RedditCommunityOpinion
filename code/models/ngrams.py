from typing import List

from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm.models import MLE
from nltk.tokenize import ToktokTokenizer
from nltk.tokenize import wordpunct_tokenize


from models.base_model import BaseModel


class NGrams(BaseModel):
    '''
    NGram language model based on sentence completion

    '''

    def __init__(self, training_set: List[str], n_param: int = 3, max_predict=4):
        super().__init__()

        '''
        Initialize the completions for the test phrase
        '''

        # convert sentence to list[words] using tokenizer
        # self.tokenizer = ToktokTokenizer()

        training_ngrams, padded_sentences = padded_everygram_pipeline(
            n_param,
            #list(map(self.tokenizer.tokenize, training_set)),
            list(map(wordpunct_tokenize, training_set)),
        )

        # print(len(training_ngrams))
        # temp = list(training_ngrams)
        # for i in range(10):
        #     print(list(temp[i]))

        self.model_obj = MLE(order=n_param)
        self.model_obj.fit(training_ngrams, padded_sentences)
        print('Vocab length: {}'.format(len(self.model_obj.vocab)))
        print('Counts: ', self.model_obj.counts)

        self.max_predict = max_predict

    def train(self, train_sentence: List[str]):
        raise NotImplementedError(
            'Train is not implemented for the NGrams model. Pass the training set in evaluate')

    def complete(self, phrase: str):
        # input_tokens = self.tokenizer.tokenize(phrase.lower())
        input_tokens = wordpunct_tokenize(phrase.lower())

        results = []
        for res_len in range(self.max_predict):
            temp = self.model_obj.generate(
                res_len+1, text_seed=input_tokens
            )

            if type(temp) == str:
                temp = [temp]

            # filter out the stop words
            temp = list(filter(lambda x: x != '</s>', temp))

            if len(temp) == 1:
                results.append(phrase + ' ' + temp[0])
            elif len(temp) > 1:
                results.append(phrase + ' ' + ' '.join(temp))

        return results


if __name__ == '__main__':
    from data.comments_loader import CommentsDataset

    obj = CommentsDataset('../dataset/small',
                          'the_donald', remove_stopwords=True)

    obj = NGrams(obj.get_training_set(), n_param=3, max_predict=10)

    # load the sample phrases
    with open('../config/questions.txt') as f:
        phrases = [x.strip() for x in f.readlines()]

    phrases = list(filter(None, phrases))

    results = dict([(x, obj.complete(x)) for x in phrases])

    for key, vals in results.items():
        print('Input Phrase: {}'.format(key))
        print('Results: ', vals)
