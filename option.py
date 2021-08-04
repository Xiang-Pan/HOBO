'''
Author: Xiang Pan
Date: 2021-07-22 04:32:22
LastEditTime: 2021-08-04 03:48:21
LastEditors: Xiang Pan
Description: 
FilePath: /HOBO/option.py
xiangpan@nyu.edu
'''
import argparse

def str2bool(str):
    return True if str.lower() == 'true' else False


parser = argparse.ArgumentParser()
parser.add_argument('--index_type', default = "IVF_FLAT",required=False,)
parser.add_argument('--op', default = "search",  required=False,)
parser.add_argument('--threshold', type = int, default = 100,  required=False,)
parser.add_argument('--wandb_log', action = "store_true")

parser.add_argument('--grid_search_config', default ="./configs/grid_search/HNSW.yaml")
# for grid search
# parser.add_argument('--build_nlist', type = int, default = 100,  required=False,)
# parser.add_argument('--search_nprobe', type = int, default = 100,  required=False,)

def get_option():
    option = parser.parse_args()
    return option
