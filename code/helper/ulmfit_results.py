import numpy as np
import pandas as pd
import fastai.text as fastai_text
from fastai.text import *


def generate_text(learn, TEXT, N_WORDS=10, N_SENTENCES=2):
    result = []

    for _ in range(N_SENTENCES):
        result.append('{}\n'.format(learn.predict(
            TEXT, N_WORDS, temperature=0.10)).encode('utf-8'))

    return result


def main():

    BASE_PATH = fastai_text.Path('../dataset/')
    PKL_PATH = BASE_PATH/'pkl-data'
    BATCH_SIZE = 32

    subreddits = [
        "aww_100000",
        "conservative_100000",
        "democrat_100000",
        "elizabethwarren_100000",
        "impeach_trump_100000",
        "libertarian_100000",
        "news_100000",
        "ourpresident_100000",
        "politics_100000",
        "sandersforpresident_100000",
        "soccer_100000",
        "the_donald_100000",
        "the_mueller_100000",
        "yangforpresidenthq_100000",
    ]

    for subreddit in subreddits:

        data = load_data(PKL_PATH, str(subreddit)+".pkl")
        learn = language_model_learner(data, AWD_LSTM, drop_mult=0.3)
        learn.load(str(subreddit)+"_stage2")
        learn.export(str(subreddit)+"_stage2.pkl")
        print("*"*10, " Results for ", subreddit, "*"*10)
        f1 = open("../config/questions_large.txt", "r")
        f2 = open("../results/ulmfit/"+str(subreddit)+"_ulmfit.txt", "wb")
        for phrase in f1:
            result = generate_text(learn, phrase)
            for entry in result:
                f2.write(entry)
            f2.write("\n".encode('utf-8'))
            f2.write("-----".encode('utf-8'))
            f2.write("\n".encode('utf-8'))
        f1.close()
        f2.close()


if __name__ == "__main__":
    main()
