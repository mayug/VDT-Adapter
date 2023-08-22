"""
Goal
---
1. Read test results from log.txt files
2. Compute mean and std across different folders (seeds)

Usage
---
Assume the output files are saved under output/my_experiment,
which contains results of different seeds, e.g.,

my_experiment/
    seed1/
        log.txt
    seed2/
        log.txt
    seed3/
        log.txt

Run the following command from the root directory:

$ python tools/parse_test_res.py output/my_experiment

Add --ci95 to the argument if you wanna get 95% confidence
interval instead of standard deviation:

$ python tools/parse_test_res.py output/my_experiment --ci95

If my_experiment/ has the following structure,

my_experiment/
    exp-1/
        seed1/
            log.txt
            ...
        seed2/
            log.txt
            ...
        seed3/
            log.txt
            ...
    exp-2/
        ...
    exp-3/
        ...

Run

$ python tools/parse_test_res.py output/my_experiment --multi-exp
"""
import re
import numpy as np
import os.path as osp
import argparse
from collections import OrderedDict, defaultdict

from dassl.utils import check_isfile, listdir_nohidden
import os
from tabulate import tabulate
import pandas as pd

def compute_ci95(res):
    return 1.96 * np.std(res) / np.sqrt(len(res))


def parse_function(*metrics, directory="", args=None, end_signal=None, logfile='log.txt'):
    print(f"Parsing files in {directory}")
    subdirs = listdir_nohidden(directory, sort=True)

    outputs = []

    for subdir in subdirs:
        # fpath = osp.join(directory, subdir, "log.txt")

        fpath = osp.join(directory, logfile)

        # print(fpath)
        # asd
        assert check_isfile(fpath)
        good_to_go = False
        output = OrderedDict()

        with open(fpath, "r") as f:
            lines = f.readlines()

            for line in lines:
                line = line.strip()

                if line == end_signal:
                    good_to_go = True

                for metric in metrics:
                    match = metric["regex"].search(line)
                    if match and good_to_go:
                        if "file" not in output:
                            output["file"] = fpath
                        num = float(match.group(1))
                        name = metric["name"]
                        output[name] = num

        if output:
            outputs.append(output)

    assert len(outputs) > 0, f"Nothing found in {directory}"

    metrics_results = defaultdict(list)

    for output in outputs:
        msg = ""
        for key, value in output.items():
            if isinstance(value, float):
                msg += f"{key}: {value:.2f}%. "
            else:
                msg += f"{key}: {value}. "
            if key != "file":
                metrics_results[key].append(value)
        print(msg)

    output_results = OrderedDict()

    print("===")
    print(f"Summary of directory: {directory}")
    for key, values in metrics_results.items():
        avg = np.mean(values)
        std = compute_ci95(values) if args.ci95 else np.std(values)
        print(f"* {key}: {avg:.2f}% +- {std:.2f}%")
        output_results[key] = avg
    print("===")

    return output_results


def main(args, end_signal):
    metric = {
        "name": args.keyword,
        "regex": re.compile(fr"\* {args.keyword}: ([\.\deE+-]+)%"),
    }

    if args.multi_exp:
        final_results = defaultdict(list)

        for directory in listdir_nohidden(args.directory, sort=True):
            directory = osp.join(args.directory, directory)
            results = parse_function(
                metric, directory=directory, args=args, end_signal=end_signal
            )

            for key, value in results.items():
                final_results[key].append(value)

        print("Average performance")
        for key, values in final_results.items():
            avg = np.mean(values)
            print(f"* {key}: {avg:.2f}%")

    else:
        results = parse_function(
            metric, directory=args.directory, args=args, end_signal=end_signal
        )
        print('results dict')
        print(results)


def get_key(log_file, key):
    # given logfile, get line matching 'SUBSAMPLE_CLASSES: ' and return the corresponding value

    with open(log_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if key in line:
                return line.split(': ')[1].strip()

def main_base2new(curr_dir, end_signal):
    metric = {
        "name": args.keyword,
        "regex": re.compile(fr"\* {args.keyword}: ([\.\deE+-]+)%"),
    }

    print(os.listdir(curr_dir))
    log_files = [i for i in os.listdir(curr_dir) if 'log' in i]

    # sort by creation time (newest first)
    log_files = sorted(log_files, key=lambda x: os.path.getctime(os.path.join(curr_dir, x)), reverse=True)
    print(log_files)
    # take the latest 2 files
    log_files = log_files[:2]
    
    # assert newest_is_new: 'newest file is not new'

    save_results = {}
    for l in log_files:

        k = get_key(os.path.join(curr_dir, l), 'SUBSAMPLE_CLASSES: ')
        try:
            results = parse_function(
                metric, directory=curr_dir, args=args, end_signal=end_signal,
                logfile= l
            )
            # print('results dict', l)
            # print(results)


            # print(['k ', k])
            # asd

            ratio = get_key(os.path.join(curr_dir, l), 'RATIO: ')
            save_results[k] = results['accuracy']
            save_results['ratio'] = ratio
            
            # print(curr_dir)
            # print([k, l])
            # print(save_results)
            # asd
        except AssertionError:
            save_results[k] = 'NA'
    return save_results


def main_all(curr_dir, end_signal):
    metric = {
        "name": args.keyword,
        "regex": re.compile(fr"\* {args.keyword}: ([\.\deE+-]+)%"),
    }

    print(os.listdir(curr_dir))
    log_files = [i for i in os.listdir(curr_dir) if 'log' in i]

    # sort by creation time (newest first)
    log_files = sorted(log_files, key=lambda x: os.path.getctime(os.path.join(curr_dir, x)), reverse=True)
    print(log_files)
    
    
    
    # assert newest_is_new: 'newest file is not new'

    save_results = {}
    for l in log_files:

        k = get_key(os.path.join(curr_dir, l), 'SUBSAMPLE_CLASSES: ')
        if k == 'all':
            try:
                results = parse_function(
                    metric, directory=curr_dir, args=args, end_signal=end_signal,
                    logfile= l
                )
                # print('results dict', l)
                # print(results)


                # print(['k ', k])
                # asd
                ratio = get_key(os.path.join(curr_dir, l), 'RATIO: ')
                save_results[k] = results['accuracy']
                save_results['ratio'] = ratio
                # print error if ratio is nan
                # print('ratio', ratio)
                # print(curr_dir)
                # assert not np.isnan(float(ratio))

                # print(curr_dir)
                # print([k, l])
                # print(save_results)
                # asd
            except AssertionError:
                save_results[k] = 'NA'
        return save_results

def main_base2new_master(encoder_dir, end_signal,filter_suffix=None):

    print(encoder_dir)
    # asd
    subdirs = os.listdir(encoder_dir)

    save_dict = {}
    print(subdirs)
    # select when 'b2n' present in all subdirs names
    subdirs = [i for i in subdirs if 'b2n' in i]

    if filter_suffix is not None:
        subdirs = [i for i in subdirs if filter_suffix in i]
    # asd
    for s in subdirs:

        curr_dir = os.path.join(encoder_dir, s)
    
        save_dict[s] = main_base2new(curr_dir, end_signal) 
    print('save dict for super directory', save_dict)
    return save_dict


def main_all_master(encoder_dir, end_signal, filter_suffix=None):

    print(encoder_dir)
    # asd
    subdirs = os.listdir(encoder_dir)

    save_dict = {}
    print(subdirs)
    # select when 'b2n' present in all subdirs names
    subdirs = [i for i in subdirs if 'all' in i]

    if filter_suffix is not None:
        subdirs = [i for i in subdirs if filter_suffix in i]

    # asd
    for s in subdirs:

        curr_dir = os.path.join(encoder_dir, s)
    
        save_dict[s] = main_all(curr_dir, end_signal) 
    print('save dict for super directory', save_dict)
    return save_dict


def get_split_df(df):
    print(df)
    new_df = pd.DataFrame(index=df.index, columns=['encoder', 'shot', 'method'] + list(df.columns))
    # print(new_df.columns)
    # asd
    encoder_name, shot_setting, method_name, ratio = 0,0,0,0
    for i in range(len(df)):
        full_name = df.index[i]
        model_class, encoder_shot, exp_name = full_name.split('--')
        print([model_class, encoder_shot, exp_name])
        
        if 'CLIP_Adapter' == model_class:
            method_name = 'clip_Adapter'
        elif 'CLIP_Adapter_gpt' == model_class:
            if 'self' in exp_name:
                method_name = 'self-attn'
            elif 'linres' in exp_name:
                method_name = 'linres'
            else:
                # continue
                raise ValueError('unknown method name')
        encoder_name, shot_setting = encoder_shot.split('_')[0], encoder_shot.split('_')[-1]
        ratio_info = exp_name.split('.')
        # print('ratio ', ratio_info)
        # ratio = ratio_info[0][-1]+ '.' + ratio_info[1][0]
        # print([encoder_name, shot_setting, method_name, ratio])
        new_df.iloc[i, 0] = encoder_name
        new_df.iloc[i, 1] = shot_setting
        new_df.iloc[i, 2] = method_name
        # new_df.iloc[i, 3] = ratio
        new_df.iloc[i, 3:] = df.iloc[i, :]
        # new_df.iloc[i, 5] = df.iloc[i, 1]
    new_df = new_df.sort_values(by=['encoder', 'shot', 'method', 'ratio'])

    return new_df


def main_dataset_master(args, end_signal):


    subdirs = ['CLIP_Adapter', 'CLIP_Adapter_gpt']

    save_dict = {}
    # select main_n based on args.all
    main_fn = main_all_master if args.all else main_base2new_master

    for s in subdirs:


        model_dir = os.path.join(args.directory, s)
        encoder_dirs = os.listdir(model_dir)
        for e in encoder_dirs:
            curr_dir = os.path.join(model_dir, e, 'nctx16_cscFalse_ctpend/seed1/')
            # if 'vit' in e:
            #     e_k = 'vit'
            # else:
            #     e_k = 'rn50'
            save_dict[s+'--'+e] = main_fn(curr_dir, end_signal, args.filter_suffix) 
    print('\n \n \n')
    print('save dict for dataset directory', save_dict)
    table = {}
    for k,v in save_dict.items():
        for i,j in v.items():
            table[k+'--'+i] = j
        # table.update(v)
    print('\n \n \n')
    # print(table)
    df = pd.DataFrame(table).transpose()
    # print(df)
    # asd
    # print(df)
    # asd
    df = get_split_df(df)
    # asd
    print('\n \n \n')

    print(tabulate(df, headers = ['new', 'base'], tablefmt="github"))

    if not args.all:
        save_path = os.path.join(args.directory, 'results_summary.csv')
        print('Saving to csv at ', save_path)
        df.to_csv(save_path)
    else:
        save_path = os.path.join(args.directory, 'results_summary_all.csv')
        print('Saving to csv at ', save_path)
        df.to_csv(save_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=str, help="path to directory")
    parser.add_argument(
        "--ci95", action="store_true", help=r"compute 95\% confidence interval"
    )
    parser.add_argument("--test-log", action="store_true", help="parse test-only logs")
    parser.add_argument(
        "--multi-exp", action="store_true", help="parse multiple experiments"
    )
    parser.add_argument(
        "--keyword", default="accuracy", type=str, help="which keyword to extract"
    )

    # argument for all or base2new
    parser.add_argument(
        "--all", action="store_true", help="parse all"
    )

    parser.add_argument(
        "--filter_suffix", default=None, type=str, help="filter suffix"
    )
    
    args = parser.parse_args()

    end_signal = "Finish training"
    if args.test_log:
        end_signal = "=> result"

    # print('args.all', args.all)
    # asd

    main_dataset_master(args, end_signal)
