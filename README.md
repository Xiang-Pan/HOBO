# HOBO

## milvus env set
```
systemctl start docker
docker run --name milvus_cpu_1.1.0 -d -p 19530:19530 -p 19121:19121 milvusdb/milvus:1.1.0-cpu-d050721-5e559c
```

## python run
```
python test.py
```