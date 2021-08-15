'''
Author: Xiang Pan
Date: 2021-07-29 17:05:12
LastEditTime: 2021-08-14 20:25:19
LastEditors: Xiang Pan
Description: 
FilePath: /HOBO/utils.py
xiangpan@nyu.edu
'''
from env import *

def sign(x, threshold):
    if x > threshold:
        return x - threshold
    else:
        return 100000 * (threshold - x)

def convert_config(args):
    # print(dict(args))
    if type(args) != dict:
        args = vars(args)
    build_keys = [b for b in args.keys() if "build" in b]
    search_keys = [b for b in args.keys() if "search" in b and b != "grid_search_config"]
    hyperparameters = dict()
    hyperparameters['index_type'] = get_index_type(args['index_type'])
    index_params = dict()
    search_params = dict()
    for b in build_keys:
        index_params[b.replace("build_","")] = args[b]
    for s in search_keys:
        search_params[s.replace("search_","")] = args[s]
    hyperparameters["index_params"] = index_params
    hyperparameters["search_params"] = search_params
    return hyperparameters



