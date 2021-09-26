import os
import time
import numpy as np
import random
import bohb.configspace as cs
import json
from milvus import Milvus, MetricType, IndexType

gDataDim = 128




class IVF_FLAT_default_build_config(object):
    def __init__(self):
        self.nlist = 2048

class IVF_FLAT_build_config(object):
    nlist =  cs.IntegerUniformHyperparameter('nlist', 1, 16384)
    configspace = cs.ConfigurationSpace([nlist], seed=123)

    # nprobe = 16
class IVF_FLAT_search_config(object):
    # TODO: UPDATE BASED ON 0.1*BUILD IF "build"
    def __init__(self, index_params):
        self.nprobe =  cs.IntegerUniformHyperparameter('nprobe', 1, int(0.1 * index_params['nlist']))
        self.configspace = cs.ConfigurationSpace([self.nprobe], seed=123)
    

class IVF_PQ_default_build_config(object):
    def __init__(self):
        self.nlist  = 2048
        self.m = 32

class IVF_PQ_build_config(object):
    nlist =  cs.IntegerUniformHyperparameter('nlist', 1, 16384)
    # M =  cs.IntegerUniformHyperparameter('M', 1, 16) #(1, ?) NEED TO DIVIDE DATA DIM!!!
    m = cs.CategoricalHyperparameter('m', [i for i in range(1,gDataDim) if gDataDim%i == 0]) # TODO: remember to modify gDataDim
    configspace = cs.ConfigurationSpace([nlist, m], seed=123)

class IVF_PQ_search_config(object):
    def __init__(self, index_params):
        self.nprobe =  cs.IntegerUniformHyperparameter('nprobe', 1, int(1 * index_params['nlist']))
        self.configspace = cs.ConfigurationSpace([self.nprobe], seed=123)
    # nprobe =  cs.IntegerUniformHyperparameter('nprobe', 1, int(16384 * 0.1))
    # configspace = cs.ConfigurationSpace([nprobe], seed=123)

class HNSW_default_build_config(object):
    def __init__(self):
        self.M = 24
        self.efConstruction = 128
    # ef = 64

class HNSW_build_config(object):
    M =  cs.IntegerUniformHyperparameter('M', 4, 64)
    efConstruction =  cs.IntegerUniformHyperparameter('efConstruction', 8, 512)
    configspace = cs.ConfigurationSpace([M, efConstruction], seed=123)

class HNSW_search_config(object):
    top_k = 100 
    ef =  cs.IntegerUniformHyperparameter('ef', top_k, 512)          # efsearch? ASK WEIZHI; (8, 512)
    configspace = cs.ConfigurationSpace([ef], seed=123)

class IVF_SQ8_default_build_config(object):
    def __init__(self):
        self.nlist = 2048
        # QuantizerType = 8
        # self.nprobe = 16

class IVF_SQ8_build_config(object):
    nlist =  cs.IntegerUniformHyperparameter('nlist', 1, 16384) #SAME AS FLAT
    configspace = cs.ConfigurationSpace([nlist], seed=123)

class IVF_SQ8_search_config(object):
    def __init__(self, index_params):
        self.nprobe =  cs.IntegerUniformHyperparameter('nprobe', 1, int(0.1 * index_params['nlist']))
        self.configspace = cs.ConfigurationSpace([self.nprobe], seed=123)
    
class IVF_FLAT_build_search_shared_config(object):
    def __init__(self):
        self.nlist =  cs.IntegerUniformHyperparameter('nlist', 1, 16384)
        self.nprobe =  cs.IntegerUniformHyperparameter('nprobe', 1, 16384)
        self.configspace = cs.ConfigurationSpace([self.nlist, self.nprobe], seed=123)


class IVF_SQ8_build_search_shared_config(object):
    def __init__(self):
        self.nlist =  cs.IntegerUniformHyperparameter('nlist', 1, 16384)
        self.nprobe =  cs.IntegerUniformHyperparameter('nprobe', 1, 16384)
        self.configspace = cs.ConfigurationSpace([self.nlist, self.nprobe], seed=123)

class IVF_PQ_build_search_shared_config(object):
    def __init__(self):
        self.nlist =  cs.IntegerUniformHyperparameter('nlist', 1, 16384)
        self.m = cs.CategoricalHyperparameter('m', [i for i in range(1,gDataDim) if gDataDim%i == 0])
        self.nprobe =  cs.IntegerUniformHyperparameter('nprobe', 1, 16384)
        self.configspace = cs.ConfigurationSpace([self.nlist, self.m, self.nprobe, ], seed=123)

class HNSW_build_search_shared_config(object):
    def __init__(self):
        self.M =  cs.IntegerUniformHyperparameter('M', 4, 64)
        self.efConstruction =  cs.IntegerUniformHyperparameter('efConstruction', 8, 512)
        top_k = 100 
        self.ef =  cs.IntegerUniformHyperparameter('ef', top_k, 512)          
        self.configspace = cs.ConfigurationSpace([self.M, self.efConstruction, self.ef], seed=123)


def get_index_type(index_type):
    if index_type == "IVF_FLAT":
        return IndexType.IVF_FLAT
    if index_type == "IVF_PQ":
        return IndexType.IVF_PQ 
    if index_type == "IVF_SQ8":
        return IndexType.IVF_SQ8
    if index_type == "HNSW":
        return IndexType.HNSW

def get_index_str(index_type):
    if index_type == IndexType.IVF_FLAT:
        return "IVF_FLAT"
    if index_type == IndexType.IVF_PQ:
        return "IVF_PQ" 
    if index_type == IndexType.IVF_SQ8:
        return "IVF_SQ8"
    if index_type == IndexType.HNSW:
        return "HNSW"



def get_default_build_config(index_type):
    if index_type == IndexType.IVF_FLAT:
        return IVF_FLAT_default_build_config().__dict__
    if index_type == IndexType.IVF_PQ:
        return IVF_PQ_default_build_config().__dict__
    if index_type == IndexType.IVF_SQ8:
        return IVF_SQ8_default_build_config().__dict__
    if index_type == IndexType.HNSW:
        return HNSW_default_build_config().__dict__

def get_search_configspace(index_type, index_params):
    if index_type == IndexType.IVF_FLAT:
        return IVF_FLAT_search_config(index_params).configspace
    if index_type == IndexType.IVF_PQ:
        return IVF_PQ_search_config(index_params).configspace
    if index_type == IndexType.IVF_SQ8:
        return IVF_SQ8_search_config(index_params).configspace
    if index_type == IndexType.HNSW:
        return HNSW_search_config().configspace

def get_default_build_configspace(index_type):
    if index_type == IndexType.IVF_FLAT:
        return IVF_FLAT_default_build_config().configspace
    if index_type == IndexType.IVF_PQ:
        return IVF_PQ_default_build_config().configspace
    if index_type == IndexType.IVF_SQ8:
        return IVF_SQ8_default_build_config().configspace
    if index_type == IndexType.HNSW:
        return HNSW_default_build_config().configspace


def get_build_configspace(index_type):
    if index_type == "IVF_FLAT" or index_type == IndexType.IVF_FLAT:
        return IVF_FLAT_build_config().configspace
    if index_type == "IVF_PQ" or index_type == IndexType.IVF_PQ:
        return IVF_PQ_build_config().configspace
    if index_type == "IVF_SQ8" or index_type == IndexType.IVF_SQ8:
        return IVF_SQ8_build_config().configspace
    if index_type == "HNSW" or index_type == IndexType.HNSW:
        return HNSW_build_config().configspace

def get_build_search_shared_configspace(index_type):
    if index_type == "IVF_FLAT" or index_type == IndexType.IVF_FLAT:
        return IVF_FLAT_build_search_shared_config().configspace
    if index_type == "IVF_PQ" or index_type == IndexType.IVF_PQ:
        return IVF_PQ_build_search_shared_config().configspace
    if index_type == "IVF_SQ8" or index_type == IndexType.IVF_SQ8:
        return IVF_SQ8_build_search_shared_config().configspace
    if index_type == "HNSW" or index_type == IndexType.HNSW:
        return HNSW_build_search_shared_config().configspace


class ENV():
    def __init__(self, args = None):
        print("ENV")
        # docker related information
        host = '127.0.0.1' 
        port = '19530'

        # get milvus client and collection_name
        self.client = Milvus(host, port)
        self.collection_name = args.collection_name
        
        # get query_vectors and set top_k
        self.query_groundtruth = self.get_groundtruth()
        self.query_vectors = self.get_query()
        self.top_k = 100

        # get status by curretn db 
        self.index_type = None
        self.index_params = None
        self.refresh_status()

        # set datadim, which is needed by some serch constraint
        global gDataDim
        gDataDim = 128

        # based on the input type, get the default build config
        if args.op == "build_params":
            self.target_index_type = get_index_type(args.index_type)
            self.target_index_params = get_default_build_config(self.target_index_type)

            is_build = False
            if self.index_type != self.target_index_type:
                is_build = True
            elif self.index_params != self.target_index_params:
                is_build = True
            
            if is_build:
                self.env_build_input(self.target_index_type ,self.target_index_params)
                self.refresh_status()

        # set search space
        self.default_build_config = get_default_build_config(self.index_type)
        self.search_configspace = get_search_configspace(self.index_type, self.index_params)
        self.build_configspace = get_build_configspace(self.index_type)
        
    def build_default_index(self):
        build_info = self.client.create_index(self.collection_name, self.index_type, self.default_build_config)

    def get_groundtruth(self):
        groundtruth = np.load("./cached_datasets/"+self.collection_name+"/"+self.collection_name+"_groundtruth.npy") 
        return groundtruth

    def get_query(self):
        query_vectors = np.load("./cached_datasets/"+self.collection_name+"/"+self.collection_name+"_query.npy")
        return query_vectors

    def get_avg_recall(self, preds, targets):
        recall_total = 0
        for i in range(len(targets)):
            recall_single = len(set(preds[i]) & set(targets[i]))
            recall_total += recall_single
        avg_recall = recall_total/len(preds)
        return avg_recall
    
    def env_build_input(self, build_type, params):
        build_info = self.client.create_index(self.collection_name, build_type, params)
        if build_info.code != 0:
            print("build_info failed!")
            print(build_info)
        self.refresh_status()
    
    def set_build_index_type(self, index_type):
        self.index_type = index_type

    def env_search_input(self, params):
        self.search_params = params
        start_time = time.time()
        status, res = self.client.search(self.collection_name, top_k = self.top_k, query_records = self.query_vectors, params = params)

        end_time = time.time()
        if status.code != 0:
            print("search failed!")
            print(status)
        query_time = (end_time - start_time)
        query_num = self.query_vectors.shape[0]
        query_per_sec = query_num/query_time
        if len(res) == 0:
            return 0, query_per_sec
        else:
            converted_res = np.zeros(res.shape)
            for i in range(len(res)):
                for j in range(len(res[i])):
                    converted_res[i][j] = res[i][j].id
            return self.get_avg_recall(converted_res, self.query_groundtruth), query_per_sec


    # base on user's input, get the target search space
    def refresh_status(self):
        """
        refresh status
        reset index_type and index_params
        reset all config space 
        """        
        status, stats = self.client.get_index_info(self.collection_name)
        self.index_type = stats._index_type
        self.index_params = stats._params

        # set config space
        self.default_build_config = get_default_build_config(self.index_type)
        self.build_configspace = get_build_configspace(self.index_type)
        self.search_configspace = get_search_configspace(self.index_type, self.index_params)


    # given full env put
    def config_input(self, config):
        # check current index type
        is_build = False

        self.refresh_status()
        if self.index_type != config['index_type']:
            is_build = True
        elif self.index_params != config['index_params']:
            is_build = True
        
        if is_build:
            self.env_build_input(config['index_type'] ,config['index_params'])
            self.refresh_status()
        

        recall, query_per_sec = self.env_search_input(config['search_params'])
        self.search_params = config['search_params']
        
        return recall, query_per_sec