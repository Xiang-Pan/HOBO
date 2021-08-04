'''
Author: Xiang Pan
Date: 2021-07-29 21:18:11
LastEditTime: 2021-08-04 03:55:15
LastEditors: Xiang Pan
Description: 
FilePath: /HOBO/grid_search.py
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

args = get_option()
config = dict()
config_file = yaml.load(open(args.grid_search_config))
name_list = [name for name in config_file['parameters']]
params_list = [config_file['parameters'][name]['values'] for name in config_file['parameters']]
config['index_type'] = config_file['index_type']['values'][0]
first = True


def input_config(config):
    global first
    global table
    if first:
        target_index_params = config['index_params']
        target_search_params = config['search_params']

        cols = ["index_type"] + list(target_index_params.keys()) + list(target_search_params.keys()) + ["recall", "query_per_sec", "loss"] 
        table = wandb.Table(columns = cols)
        first = False
    wandb.log(config['index_params'])
    wandb.log(config['search_params'])

    recall , query_per_sec = env.config_input(config)
    threshold = 95
    loss = sign(recall, threshold) + query_per_sec

    wandb.log({"recall": recall})
    wandb.log({"query_per_sec": query_per_sec})
    wandb.log({"loss": loss})

    data = [str(config['index_type'])] +list(env.index_params.values()) + list(env.search_params.values()) + [recall, query_per_sec, loss]
    table.add_data(*data)
    

def dfs(a, l, depth):
    if len(a) == 1: # end condition
        t = a[0]
        if type(t) == str:
            for key in config.keys():
                if key in t:
                    t = t.replace(key, str(config[key]))
            t = eval(t)
        for i in t:
            l.append(i)
            config[name_list[depth]] = i
            converted_config = convert_config(config)
            # print(converted_config)
            input_config(converted_config)
            l.remove(i)
    else:
        t = a[0]
        if type(t) == str:
            t = eval(t)
        for i in t:
            l.append(i)
            config[name_list[depth]] = i
            for j in a[1:]:
                dfs(a[1:],l, depth+1)
            l.remove(i)

if __name__ == "__main__":

    # args.index_type = 'HNSW'
    # args.build_M = -1
    # args.build_efConstruction = -1
    # args.search_ef = -1

    # name = get_exp_name(args)
    run = wandb.init()

    wandb.run.name = args.grid_search_config

    # config = convert_config(args)
    run = wandb.init()
    env = ENV()

    dfs(params_list,[],0)
    
    
    # for M in range(4,64,10):
        # for efConstruction in range(8,512,50):
            # for ef in range(100,32768,100):
                # config['index_params']['M'] = M
                # config['index_params']['efConstruction'] = efConstruction
                # config['search_params']['ef'] = ef

                # wandb.log(config['index_params'])
                # wandb.log(config['search_params'])

                
                # recall , query_per_sec = env.config_input(config)
                # threshold = 95
                # loss = sign(recall, threshold) + query_per_sec

                # wandb.log({"recall": recall})
                # wandb.log({"query_per_sec": query_per_sec})
                # wandb.log({"loss": loss})

                # data = [str(config['index_type'])] +list(env.index_params.values()) + list(env.search_params.values()) + [recall, query_per_sec, loss]
                # table.add_data(*data)
            
    run.log({str(config['index_type']):table})
