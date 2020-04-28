'''
Script to convert txt results to csv
'''
import csv
import glob
import os
import sys

import pandas as pd


def parse_gpt2_file(filename: str):
    '''
    Parses a GPT2 file to return all the results
    '''

    results = []
    curr_output = ''

    with open(filename, 'r') as f:

        for line in f.readlines():
            if len(line) == 0:
                continue
            if line.startswith('-------------'):
                if len(curr_output) > 0:
                    curr_output = curr_output.strip()
                    curr_output = curr_output.replace(
                        '\n', ' ').replace('\r', '')
                    results.append(curr_output)
                    curr_output = ''
            else:
                curr_output += line

    if len(curr_output) > 0:
        curr_output = curr_output.strip()
        curr_output = curr_output.replace(
            '\n', ' ').replace('\r', '')
        results.append(curr_output)
        curr_output = ''

    return results


def parse_ulmfit_file(filename: str):
    '''
    Parses a GPT2 file to return all the results
    '''

    results = []
    curr_output = ''

    with open(filename, 'r') as f:

        for line in f.readlines():
            if len(line) == 0:
                continue
            if line.startswith('-----'):
                if len(curr_output) > 0:
                    curr_output = curr_output.strip()
                    curr_output = curr_output.replace(
                        '\n', ' ').replace('\r', '')
                    results.append(curr_output)
                    curr_output = ''
            else:
                curr_output += line

    if len(curr_output) > 0:
        curr_output = curr_output.strip()
        curr_output = curr_output.replace(
            '\n', ' ').replace('\r', '')
        results.append(curr_output)
        curr_output = ''

    return results


if __name__ == '__main__':
    args = sys.argv[1:]

    mode = args[0]

    if not (mode == 'gpt2' or mode == 'ulmfit'):
        raise NotImplementedError('This mode is not implemented')

    dirname = args[1]

    filenames = glob.glob(os.path.join(dirname, '*.txt'))

    for filename in filenames:
        if mode == 'gpt2':
            results = parse_gpt2_file(filename)
        elif mode == 'ulmfit':
            results = parse_ulmfit_file(filename)

        # figure out the outpot name
        result_name = os.path.splitext(filename)[0] + '.csv'
        resultsmall_name = os.path.splitext(filename)[0] + '_small.csv'

        df = pd.DataFrame(results, columns=['output'])

        df['usable'] = False

        df.to_csv(result_name, quoting=csv.QUOTE_MINIMAL)

        # df_small = df.iloc[::5, :]
        # df_small.to_csv(resultsmall_name, quoting=csv.QUOTE_MINIMAL)
