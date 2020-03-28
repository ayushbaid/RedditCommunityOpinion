from helper.wlm_lstm_main import WLMRunner

if __name__ == '__main__':
    # load the subreddit names

    with open('../config/subreddits.txt') as f:
        subreddits = [x.strip() for x in f.readlines()]

    subreddits = filter(None, subreddits)

    for subreddit in subreddits:
        print('Running {}'.format(subreddit))
        obj = WLMRunner(subreddit, load_from_disk=True)

        result_file = open(
            '../results/{}_completion.txt'.format(subreddit), 'w')
        result_file.write('sentence,log_probability\n')

        phrase_ids = obj.phrase_ids

        for inp_ids in obj.phrase_ids:
            inp_str = [obj.corpus.lookup_word(id) for id in inp_ids]

            results = obj.beam_search([(inp_str, 0.0)], num_entries=5)

            for sentence, log_prob in results:
                result_file.write(' '.join(sentence) +
                                  ',{}\n'.format(log_prob))

        result_file.close()
