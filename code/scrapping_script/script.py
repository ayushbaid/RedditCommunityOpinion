import datetime as dt
import sys
import os

import praw
import numpy as np
from psaw import PushshiftAPI

api = None


def get_n_comments(n, subreddit_name, start_epoch, end_epoch):
    """
    args:
        n: number of comments to return
        subreddit_name: name of subreddit
        dr: date after; look at pushshift.io api for usage info

    returns:
        cache: first n comments of the subbredit's hot threads.
        if the thread had less than n comments, it return all of them

    """
    gen = api.search_comments(subreddit=subreddit_name,
                              after=start_epoch,
                              before=end_epoch,
                              limit=n)

    max_response_cache = n
    cache = []
    i = 0
    for c in gen:
        if c.body == "[removed]" or c.body == "[deleted]":
            continue

        i = i+1
        if i % 1000 == 0:
            print(i)

        cache.append(c.body)

        if i >= max_response_cache:
            break
    return cache


def wrapper(n, subreddit_name):
    """
    args:
        n: number of comments to return
        subreddit_name: name of subreddit

    returns:
        commentList: first n comments of the subbredit's hot threads
        as np array

    descriptions: 
        saves the first n comments of a subthread to a numpy file
        titled: [subreddit_name]_n.npy
    """
    # fetch submission across months
    dates = [
        (int(dt.datetime(2019, x, 1).timestamp()),
         int(dt.datetime(2019, x, 28).timestamp())
         ) for x in range(12, 0, -1)
    ]

    # we will try to spread the n across 4 months; if needed use other months

    num_comments_per_month = n//4

    total_len = 0
    for idx, (start_epoch, end_epoch) in enumerate(dates):
        if total_len >= n:
            break

        train_path = '../dataset/{}_{}_train_part{}.npy'.format(
            subreddit_name, n, idx)
        val_path = '../dataset/{}_{}_val_part{}.npy'.format(
            subreddit_name, n, idx)

        if os.path.exists(val_path):
            # skip download if already exists
            temp = np.load(val_path)

            total_len += temp.shape[0]
            temp = None

            continue

        # check if file already exists

        comments = get_n_comments(
            num_comments_per_month, subreddit_name, start_epoch, end_epoch)

        curr_len = len(comments)
        total_len += curr_len

        train_len = int(0.7*curr_len)

        comments_array = np.array(comments)
        np.random.shuffle(comments_array)

        # split into train and val using 70-30 split
        np.save(
            train_path,
            comments_array[:train_len]
        )

        np.save(
            val_path,
            comments_array[train_len:]
        )


if __name__ == '__main__':
    argv = sys.argv[1:]

    reddit = praw.Reddit(client_id='PblUd0Q2RvF1oA',
                         client_secret='5VM1zqp0jk_DfEpAH4UnpK1Lb4M',
                         password='Cricketboy@008',
                         user_agent='wstm',
                         username='tycoblaster3000')

    api = PushshiftAPI(reddit)

    num_comments = int(argv[0])

    subreddits = argv[1:]

for subreddit in subreddits:
    print(subreddit)
    wrapper(num_comments, subreddit)
