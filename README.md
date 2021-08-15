<!--
 * @Author: Xiang Pan
 * @Date: 2021-07-10 00:23:34
 * @LastEditTime: 2021-08-14 20:10:09
 * @LastEditors: Xiang Pan
 * @Description: 
 * @FilePath: /HOBO/README.md
 * xiangpan@nyu.edu
-->
# HOBO

## milvus env set
```
systemctl start docker
docker run --name milvus_cpu_1.1.0 -d -p 19530:19530 -p 19121:19121 milvusdb/milvus:1.1.0-cpu-d050721-5e559c
docker restart 6e7513d3203b  
```

## supported dataset
siftsmall
sift


## supported index_type 
```
IVF_FLAT
IVF_PQ
IVF_SQ8
HNSW
```

## python run
```
python bohb_op.py --op build_type --threshold 95 --wandb_log
python bohb_op.py --index_type IVF_FLAT --op build_params --threshold 95 --wandb_log
python bohb_op.py --index_type IVF_FLAT --op search_params --threshold 95 --wandb_log
```

## monitor setup
```
metric:
  enable: true                 # Set the value to true to enable the Prometheus monitor.
  address: <your_IP_address>   # Set the IP address of Pushgateway.
  port: 9091                   # Set the port number of Pushgateway.
```

# Grid Search 
## ANN_SIFT10K
HNSW     https://wandb.ai/xiang-pan/HOBO/runs/3vdvm6gs  
IVF_SQ8  https://wandb.ai/xiang-pan/HOBO/runs/26z6cea5  (run locally, need rerun)  
IVF_FLAT https://wandb.ai/xiang-pan/HOBO/runs/22n2lk07  
IVF_PQ  https://wandb.ai/xiang-pan/HOBO/runs/22mj5iiv  

# TODO
TODO :How to implement an auto grid search tool


# Benchmark
![image](https://raw.githubusercontent.com/matsui528/annbench_leaderboard/main/result_img/2021_02_23/deep1m.png)


# Updata Logs