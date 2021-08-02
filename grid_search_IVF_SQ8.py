'''
Author: Xiang Pan
Date: 2021-07-29 21:18:11
LastEditTime: 2021-08-02 19:25:07
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

    run = wandb.init()
    wandb.run.name = "IVF_SQ8_gird_search"

    config = convert_config(args)
    run = wandb.init()
    env = ENV()

    target_index_params = config['index_params']
    target_search_params = config['search_params']

    cols = ["index_type"] + list(target_index_params.keys()) + list(target_search_params.keys()) + ["recall", "query_per_sec", "loss"] 
    table = wandb.Table(columns = cols)    

    cols = ["index_type"] + list(env.index_params.keys()) + list(env.search_params.keys()) + ["recall", "query_per_sec", "loss"] 
    table = wandb.Table(columns = cols)    

    for nlist in range(1,16384,100):
        for nprobe in range(1, int(0.1*nlist),100):
            config['index_params']['nlist'] = nlist
            config['search_params']['nprobe'] = nprobe
            
            recall , query_per_sec = env.config_input(config)
            threshold = 95
            loss = sign(recall, threshold) + query_per_sec

            wandb.log(config['index_params'])
            wandb.log(config['search_params'])
            wandb.log({"recall": recall})
            wandb.log({"query_per_sec": query_per_sec})
            wandb.log({"loss": loss})

            data = [str(config['index_type'])] +list(env.index_params.values()) + list(env.search_params.values()) + [recall, query_per_sec, loss]
            table.add_data(*data)
            
    run.log({str(config['index_type']):table})
