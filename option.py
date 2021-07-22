'''
Author: Xiang Pan
Date: 2021-07-22 04:32:22
LastEditTime: 2021-07-22 06:57:07
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



def get_option():
    option = parser.parse_args()
    return option
