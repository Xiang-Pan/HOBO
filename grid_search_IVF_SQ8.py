'''
Author: Xiang Pan
Date: 2021-07-29 21:18:11
LastEditTime: 2021-08-02 15:36:27
LastEditors: Xiang Pan
Description: 
FilePath: /HOBO/grid_search_IVF_SQ8.py
xiangpan@nyu.edu
'''
import wandb
from milvus import IndexType
from env import *
from option import *
import sys
import yaml
from utils import *

# class IVF_SQ8_build_config(object):
#     nlist =  cs.IntegerUniformHyperparameter('nlist', 1, 16384) #SAME AS FLAT
#     configspace = cs.ConfigurationSpace([nlist], seed=123)

# class IVF_SQ8_search_config(object):
#     nprobe =  cs.IntegerUniformHyperparameter('nprobe', 0, 2048)
#     configspace = cs.ConfigurationSpace([nprobe], seed=123)
    


if __name__ == "__main__":
    args = get_option()

    args.index_type = 'IVF_SQ8'
    args.build_nlist = -1
    args.search_nprobe = -1

    config = convert_config(args)
    run = wandb.init()
    env = ENV()
    # env.build_params = config['build_params'] # 
    env.search_params = config['search_params']

    cols = ["index_type"] + list(env.index_params.keys()) + list(env.search_params.keys()) + ["recall", "query_per_sec", "loss"] 
    table = wandb.Table(columns = cols)    

    for nlist in range(1,1024,10):
        for nprobe in range(1, 0.1*nlist,10):
            config['index_params']['nlist'] = 16384
            config['search_params']['nprobe'] = nprobe
            
            recall , query_per_sec = env.config_input(config)
            threshold = 95
            loss = sign(recall, threshold) + query_per_sec

            wandb.log({"recall": recall})
            wandb.log({"query_per_sec": query_per_sec})
            wandb.log({"loss": loss})

            data = [str(config['index_type'])] +list(env.index_params.values()) + list(env.search_params.values()) + [recall, query_per_sec, loss]
            table.add_data(*data)
            
    run.log({str(config['index_type']):table})
