# from fastai.text import *
import glob
import os
import zipfile

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
        if 'val' in name: df['is_valid'] = True
        else: df['is_valid'] = False
        full_df = full_df.append(df)

    full_df.reset_index(inplace=True,drop=True)
    full_df.to_csv(CSV_PATH/str(subreddit+".csv"))
    return
  
  def create_save_databunch_pkl(self,subreddit, batch_size, CSV_PATH, PKL_PATH):
    data = (TextList.from_csv(CSV_PATH, str(subreddit+".csv"), cols='Comments')
                .split_from_df(col='is_valid')
                .label_for_lm()
                .databunch(bs=batch_size))
    temp = data.path
    data.path = PKL_PATH
    data.save(str(subreddit+".pkl"))
    data.path = temp
    return


def main():

  BASE_PATH = Path('../dataset/100k')
  ZIP_PATH = BASE_PATH/'zip-data'
  UNZIP_PATH = BASE_PATH/'unzip-data'
  CSV_PATH = BASE_PATH/'csv-data'
  PKL_PATH = BASE_PATH/'pkl-data'

  BATCH_SIZE = 32

  subreddits = ['news_100k'] #'news_100k'

  data_pre = DataProcessing()

  for subreddit in subreddits:
    print ("*"*50, subreddit, "Pre-processing", "*"*50)

    print ("Step 1/4 - Unzipping {}...".format(subreddit))
    # %time data_pre.unzip_data(ZIP_PATH/str(subreddit+".zip"), UNZIP_PATH)

    print ("Step 2/4 - Extracting all filenames from unzipped folder...")
    # subreddit_fnames = data_pre.get_filenames(UNZIP_PATH/subreddit, '*.npy')
    # for name in subreddit_fnames: print (name)

    print ("Step 3/4 - Combining files to create data csv...")
    # %time data_pre.create_csv(subreddit, subreddit_fnames, CSV_PATH)

    print ("Step 4/4 - Creating and Saving databunch from csv...")
    # %time data_pre.create_save_databunch_pkl(subreddit, BATCH_SIZE, CSV_PATH, PKL_PATH)

if __name__ == "__main__":
    main()