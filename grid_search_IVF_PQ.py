'''
Author: Xiang Pan
Date: 2021-07-29 21:18:11
LastEditTime: 2021-08-14 20:27:33
LastEditors: Xiang Pan
Description: 
FilePath: /HOBO/grid_search_IVF_PQ.py
xiangpan@nyu.edu
'''
import wandb
from milvus import IndexType
from env import *
from option import *
import sys
import yaml
from utils import *

# class IVF_PQ_build_config(object):
#     nlist =  cs.IntegerUniformHyperparameter('nlist', 0, 16384)
#     # M =  cs.IntegerUniformHyperparameter('M', 1, 16) #(1, ?) NEED TO DIVIDE DATA DIM!!!
#     M = cs.CategoricalHyperparameter('M', [i for i in range(1,16) if gDataDim%i == 0]) # TODO: remember to modify gDataDim
#     configspace = cs.ConfigurationSpace([nlist, M], seed=123)

# class IVF_PQ_search_config(object):
#     nprobe =  cs.IntegerUniformHyperparameter('nprobe', 0, 20)
#     configspace = cs.ConfigurationSpace([nprobe], seed=123)


if __name__ == "__main__":
    args = get_option()

    args.index_type = 'IVF_PQ'
    args.build_nlist = -1
    args.build_m = -1
    args.search_nprobe = -1

    run = wandb.init()
    wandb.run.name = "IVF_PQ_gird_search"

    config = convert_config(args)
    run = wandb.init()
    env = ENV(args)
    

    target_index_params = config['index_params']
    target_search_params = config['search_params']

    cols = ["index_type"] + list(target_index_params.keys()) + list(target_search_params.keys()) + ["recall", "query_per_sec", "loss"] 
    table = wandb.Table(columns = cols)     

    for nlist in range(1,16384,100):
        for m in [i for i in range(1,16) if gDataDim%i == 0]:
            for nprobe in [1, max(1,int(0.1*nlist)), 100]:
                config['index_params']['nlist'] = nlist
                # config['index_params']['M'] = M
                config['index_params']['m'] = m
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
