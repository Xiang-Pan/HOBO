'''
Author: Xiang Pan
Date: 2021-07-29 21:18:11
LastEditTime: 2021-08-02 15:37:35
LastEditors: Xiang Pan
Description: 
FilePath: /HOBO/grid_search_HNSW.py
xiangpan@nyu.edu
'''
import wandb
from milvus import IndexType
from env import *
from option import *
import sys
import yaml
from utils import *

# class HNSW_build_config(object):
#     M =  cs.IntegerUniformHyperparameter('M', 4, 64)
#     efConstruction =  cs.IntegerUniformHyperparameter('efConstruction', 8, 512)
#     configspace = cs.ConfigurationSpace([M, efConstruction], seed=123)

# class HNSW_search_config(object):
#     top_k = 100 #trace it back!!
#     # nprobe =  cs.IntegerUniformHyperparameter('nprobe', 8, 512)# efsearch? ASK WEIZHI; (8, 512)
#     ef =  cs.IntegerUniformHyperparameter('ef', top_k, 512)          # efsearch? ASK WEIZHI; (8, 512)
#     configspace = cs.ConfigurationSpace([ef], seed=123)


if __name__ == "__main__":
    args = get_option()
    args.index_type = 'HNSW'
    args.build_M = -1
    args.build_efConstruction = -1
    args.search_ef = -1

    # name = get_exp_name(args)
    run = wandb.init()
    wandb.run.name = "HNSW_gird_search"

    config = convert_config(args)
    run = wandb.init()
    env = ENV()
    

    target_index_params = config['index_params']
    target_search_params = config['search_params']
    print(config)

    cols = ["index_type"] + list(target_index_params.keys()) + list(target_search_params.keys()) + ["recall", "query_per_sec", "loss"] 
    table = wandb.Table(columns = cols)    
    
    for M in range(4,64,10):
        for efConstruction in range(8,512,10):
            for ef in range(100,512,10):
                config['index_params']['M'] = M
                config['index_params']['efConstruction'] = efConstruction
                config['search_params']['ef'] = ef
                
                recall , query_per_sec = env.config_input(config)
                threshold = 95
                loss = sign(recall, threshold) + query_per_sec

                wandb.log({"recall": recall})
                wandb.log({"query_per_sec": query_per_sec})
                wandb.log({"loss": loss})

                data = [str(config['index_type'])] +list(env.index_params.values()) + list(env.search_params.values()) + [recall, query_per_sec, loss]
                table.add_data(*data)
            
    run.log({str(config['index_type']):table})
