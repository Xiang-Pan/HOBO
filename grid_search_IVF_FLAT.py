'''
Author: Xiang Pan
Date: 2021-07-29 21:18:11
LastEditTime: 2021-08-02 15:41:41
LastEditors: Xiang Pan
Description: 
FilePath: /HOBO/grid_search_IVF_FLAT.py
xiangpan@nyu.edu
'''
import wandb
from milvus import IndexType
from env import *
from option import *
import sys
import yaml
from utils import *



if __name__ == "__main__":
    args = get_option()
    
    args.index_type = 'IVF_FLAT'
    args.build_nlist = -1
    args.search_nprobe = -1

    run = wandb.init()
    wandb.run.name = "IVF_FLAT_gird_search"

    config = convert_config(args)
    run = wandb.init()
    env = ENV()
    # env.build_params = config['build_params'] # 
    env.search_params = config['search_params']

    cols = ["index_type"] + list(env.index_params.keys()) + list(env.search_params.keys()) + ["recall", "query_per_sec", "loss"] 
    table = wandb.Table(columns = cols)    

    for i in range(1,16384,10):
        for j in range(1, int(0.1*i),10):
            config['index_params']['nlist'] = i
            config['search_params']['nprobe'] = j
            
            recall , query_per_sec = env.config_input(config)
            threshold = 95
            loss = sign(recall, threshold) + query_per_sec

            wandb.log({"recall": recall})
            wandb.log({"query_per_sec": query_per_sec})
            wandb.log({"loss": loss})

            data = [str(config['index_type'])] +list(env.index_params.values()) + list(env.search_params.values()) + [recall, query_per_sec, loss]
            table.add_data(*data)
            
    run.log({str(config['index_type']):table})
