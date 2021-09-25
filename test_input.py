'''
Author: Xiang Pan
Date: 2021-08-14 19:05:46
LastEditTime: 2021-09-25 18:08:09
LastEditors: Xiang Pan
Description: 
FilePath: /HOBO/test_input.py
xiangpan@nyu.edu
'''
from utils import *
from option import *
import yaml

# def input_config():
    # config_file = yaml.load(open("./configs/sample_input/IVF_PQ.yaml"))
    # config['index_type'] = config_file['index_type']['values'][0]
    # print(config_file)
    # config['index_params'] =
    # config['search_params'] = 
    # converted_config = convert_config(config)
    # name_list = [name for name in config_file['parameters']]
    # params_list = [config_file['parameters'][name]['values'] for name in config_file['parameters']]
    # config['index_type'] = config_file['index_type']['values'][0]
    # config['index_params'] = config_file['parameters'][]
    # global first
    # global table
    # if first:
    #     target_index_params = config['index_params']
    #     target_search_params = config['search_params']

    #     cols = ["index_type"] + list(target_index_params.keys()) + list(target_search_params.keys()) + ["recall", "query_per_sec", "loss"] 
    #     table = wandb.Table(columns = cols)
    #     first = False
    # wandb.log(config['index_params'])
    # wandb.log(config['search_params'])
    # args = get_option()
    # env = ENV(args)
    # recall , query_per_sec = env.config_input(config)
    # threshold = 95
    # loss = sign(recall, threshold) + query_per_sec


def test_input():
    args = get_option()
    env = ENV(args)
    config = dict()

    config_file = yaml.load(open("./configs/sample_input/IVF_PQ.yaml"), Loader=yaml.FullLoader)
    config['index_type'] = get_index_type(config_file['index_type']['values'][0])
    config['index_params'] = config_file['parameters']['index_params']
    config['search_params'] = config_file['parameters']['search_params']

    recall , query_per_sec = env.config_input(config)
    threshold = 95
    loss = sign(recall, threshold) + query_per_sec
    print(recall, query_per_sec, loss)
    return 


if __name__ == "__main__":
    test_input()