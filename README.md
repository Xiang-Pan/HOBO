<!--
 * @Author: Xiang Pan
 * @Date: 2021-07-10 00:23:34
 * @LastEditTime: 2021-07-22 16:52:54
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

## supported index_type 
```
IVF_FLAT
IVF_PQ
IVF_SQ8
HNSW
```

## python run
```
python test.py --index_type IVF_FLAT --op build
```

## monitor setup
```
metric:
  enable: true                 # Set the value to true to enable the Prometheus monitor.
  address: <your_IP_address>   # Set the IP address of Pushgateway.
  port: 9091                   # Set the port number of Pushgateway.
```