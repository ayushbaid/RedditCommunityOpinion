from fastai.text import *


class SubredditULMFit(object):

    def load_databunch(self, subreddit, PKL_PATH, BATCH_SIZE):
        data = load_data(PKL_PATH, str(subreddit + ".pkl"), bs=BATCH_SIZE)
        return data


def main():

    # Path('drive/My Drive/Google Colab/wstm-project/100k/')
    BASE_PATH = Path('../dataset')

    RESULTS_PATH = Path('../results/ulmfit')
    # ZIP_PATH = BASE_PATH/'zip-data'
    UNZIP_PATH = BASE_PATH  # BASE_PATH/'unzip-data'
    CSV_PATH = BASE_PATH/'csv-data'
    PKL_PATH = BASE_PATH/'pkl-data'
    IMG_PATH = RESULTS_PATH/'img-data'

    BATCH_SIZE = 64

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
    reddit_ulmfit = SubredditULMFit()
    finding_lr = False

    lr = 3e-2

    for subreddit in subreddits:
        print("*"*50, subreddit, "Training", "*"*50)

        print("Step 1/7 - Loading DataBunch {}...".format(subreddit))
        data = reddit_ulmfit.load_databunch(subreddit, PKL_PATH, BATCH_SIZE)

        print("Step 2/7 - Loading pre-trained wikitext language model...")
        learn = language_model_learner(data, AWD_LSTM, drop_mult=0.3)

        print("Step 3/7 - Observe best learning rate...")
        learn.lr_find()

        print("Step 4/7 - Saving figure for learning rate...")
        fig = learn.recorder.plot(return_fig=True)
        fig.savefig(IMG_PATH/str(subreddit + ".png"))

        if (finding_lr):
            print("Exiting at Step 4...")
            continue

        print("Step 4/7 - Fit data to only last few layers")
        learn.fit_one_cycle(1, lr, moms=(0.8, 0.7))

        print("Step 5/7 - Saving stage 1 model")
        learn.save(str(subreddit + '_stage1'))

        print("Step 6/7 - Unfreeze the network and fit data to all layers")
        learn.unfreeze()
        learn.fit_one_cycle(15, 1e-3, moms=(0.8, 0.7))

        print("Step 7/7 - Saving stage 2 model")
        learn.save(str(subreddit + '_stage2'))


if __name__ == "__main__":
    main()
