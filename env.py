import os
import time
import numpy as np
import random
from milvus import Milvus, MetricType, IndexType

# SERVER_ADDR = '127.0.0.1'
# SERVER_PORT = '19530'





class ENV():
    def __init__(self):
        print("ENV")
        host = '127.0.0.1'
        port = '19530'
        self.client = Milvus(host, port)
        self.collection_name = 'siftsmall'
        self.query_groundtruth = self.get_groundtruth()
        self.query_vectors = self.get_query()
        self.top_k = 100


    def get_groundtruth(self):
        groundtruth = np.load("./cached_datasets/siftsmall_numpy/siftsmall_groundtruth.npy") #! NEED modified to name-based
        return groundtruth

    def get_query(self):
        siftsmall_query = np.load("./cached_datasets/siftsmall_numpy/siftsmall_query.npy")
        query_vectors = siftsmall_query
        return query_vectors

    def get_avg_recall(self, preds, targets):
        recall_total = 0
        for i in range(len(targets)):
            recall_single = len(set(preds[i]) & set(targets[i]))
            recall_total += recall_single
        avg_recall = recall_total/len(preds)
        return avg_recall


    def env_input(self, params):
        statue, res = self.client.search(self.collection_name, top_k = self.top_k, query_records = self.query_vectors, params = params)
        converted_res = np.zeros(res.shape)
        for i in range(len(res)):
            for j in range(len(res[i])):
                converted_res[i][j] = res[i][j].id
        return self.get_avg_recall(converted_res, self.query_groundtruth)

        