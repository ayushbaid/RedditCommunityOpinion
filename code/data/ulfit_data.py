import glob
import os
import zipfile

import numpy as np
import pandas as pd
import fastai.text as fastai_text


class DataProcessing(object):

    def get_filenames(self, fldr_path, filetype):
        path = str(fldr_path/filetype)
        return glob.glob(path)

    def unzip_data(self, src_fldr, tgt_fldr):
        with zipfile.ZipFile(src_fldr, "r") as zip_ref:
            zip_ref.extractall(tgt_fldr)

    def create_csv(self, subreddit, subreddit_fnames, CSV_PATH):
        full_df = pd.DataFrame()
        for filename in subreddit_fnames:
            name = os.path.splitext(os.path.basename(filename))[0]
            np_data = np.load(filename)
            df = pd.DataFrame(np_data, columns=["Comments"])
            if 'val' in name:
                df['is_valid'] = True
            else:
                df['is_valid'] = False
            full_df = full_df.append(df)

        print("Before removing NaNs - ", full_df.shape)
        full_df = full_df[full_df['Comments'] != '']
        full_df.dropna(inplace=True)
        print("After removing NaNs - ", full_df.shape)
        full_df.reset_index(inplace=True, drop=True)
        full_df.to_csv(CSV_PATH/str(subreddit+".csv"))
        return

    def create_save_databunch_pkl(self, subreddit, batch_size, CSV_PATH, PKL_PATH):
        data = (fastai_text.TextList.from_csv(CSV_PATH, str(subreddit+".csv"), cols='Comments')
                .split_from_df(col='is_valid')
                .label_for_lm()
                .databunch(bs=batch_size))
        temp = data.path
        data.path = PKL_PATH
        data.save(str(subreddit+".pkl"))
        data.path = temp
        return


def main():

    BASE_PATH = fastai_text.Path('../dataset/')
    ZIP_PATH = BASE_PATH/'zip-data'
    # UNZIP_PATH = BASE_PATH/'unzip-data'
    UNZIP_PATH = BASE_PATH
    CSV_PATH = BASE_PATH/'csv-data'
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

    data_pre = DataProcessing()

    for subreddit in subreddits:
        print("*"*50, subreddit, "Pre-processing", "*"*50)

        print("Step 1/4 - Unzipping {}...".format(subreddit))
        # %time data_pre.unzip_data(ZIP_PATH/str(subreddit+".zip"), UNZIP_PATH)

        print("Step 2/4 - Extracting all filenames from unzipped folder...")
        subreddit_fnames = data_pre.get_filenames(
            UNZIP_PATH, subreddit+'*.npy')
        for name in subreddit_fnames:
            print(name)

        print("Step 3/4 - Combining files to create data csv...")
        data_pre.create_csv(subreddit, subreddit_fnames, CSV_PATH)

        print("Step 4/4 - Creating and Saving databunch from csv...")
        data_pre.create_save_databunch_pkl(
            subreddit, BATCH_SIZE, CSV_PATH, PKL_PATH)


if __name__ == "__main__":
    main()
