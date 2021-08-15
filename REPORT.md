<!--
 * @Author: Xiang Pan
 * @Date: 2021-08-13 18:00:59
 * @LastEditTime: 2021-08-14 21:24:12
 * @LastEditors: Xiang Pan
 * @Description: 
 * @FilePath: /HOBO/REPORT.md
 * xiangpan@nyu.edu
-->

**Threshold 95**
# Introduction
For solving the optimization of milvus hyperparameters, we use the Bayesian Optimization and Hyperband(BOHB)[<sup>1</sup>](#refer-anchor-1) as our parameter search method.

# Implementation
There are several level hyperparameters in milvus, including index_type, index_params and search_params.
To get an end-to-end solution for index, we use BOHB in different level. For Index Type, to eminent the randomness of 


<!-- # Environment
<!-- Note: Grid Search is done using zilliz server and BO is based on my local machine. -->
 -->

# Hardwareware Information


# Index Type Optimization

## 

| Method | index_type | M | efConstruction | ef  | recall | query_per_sec      | loss                |
|--------|------------|---|----------------|-----|--------|--------------------|---------------------|
| BOHB(Index Type Loop)   | 'HNSW'     | 9 | 255            | 116 | 98.71  | 20035.845992165854 | -20032.135992165855 |
| BOHB(Index Type BO)   | 'HNSW'     | 9 | 255            | 116 | 98.71  | 20035.845992165854 | -20032.135992165855 |


|             | index_type | nlist | M | nprobe | recall | query_per_sec | loss          |
|-------------|------------|-------|---|--------|--------|---------------|---------------|
| [Grid Search](https://wandb.ai/xiang-pan/HOBO/runs/3vdvm6gs)| 'HNSW'     | 4  | 158            | 200 | 97.11  | 18331.748252       | -18329.638252       |



# Index Parameters Optimization


## IVF_FLAT

|             | index_type | nlist | nprobe | recall | query_per_sec      | loss                |
|-------------|------------|-------|--------|--------|--------------------|---------------------|
| [BOHB](https://wandb.ai/xiang-pan/HOBO/runs/9ughlu3l)        |    2883 |       54 |    99.68 |           14911 | -14906.3 |
| [Grid Search](https://wandb.ai/xiang-pan/HOBO/runs/22n2lk07) | 'IVF_FLAT' | 14601 | 101    | 100.0  | 14402.032758       | -14397.032758       |

## IVF_SQ8
|             | index_type | nlist | nprobe | recall | query_per_sec     | loss               |
|-------------|------------|-------|--------|--------|-------------------|--------------------|
| [BOHB](https://wandb.ai/xiang-pan/HOBO/runs/2hnt39qn)    | 'IVF_SQ8'  |    8405 |       46 |    98.86 |         13827.5 | -13823.7 |
| [Grid Search](https://wandb.ai/xiang-pan/HOBO/runs/26z6cea5) | 'IVF_SQ8'  | 5401  | 101    | 99.49  | 13080.62997       | -13076.13997       |

## IVF_PQ
|                                                              | index_type | m | nlist | nprobe | recall | query_per_sec | loss           |
|--------------------------------------------------------------|------------|---|-------|--------|--------|---------------|----------------|
| [BOHB](https://wandb.ai/xiang-pan/HOBO/runs/2hh95hjr)         | 'IVF_PQ'   | 8 | 3800  | 205    | 70.69  | 10549.1       | 2.42045e+06    |
| [Grid Search](https://wandb.ai/xiang-pan/HOBO/runs/2i7nos9y) | 'IVF_PQ'   | 8 | 8001  | 800    | 70.69  | 1733.677784   | 2432733.677784 |


## HNSW
| Method      | index_type | M  | efConstruction | ef  | recall | query_per_sec      | loss                |
|-------------|------------|----|----------------|-----|--------|--------------------|---------------------|
| [BOHB](https://wandb.ai/xiang-pan/HOBO/runs/1gkilnbh)        | 'HNSW'     |  18 |               92 |  157 |    99.85 |         17868.6 | -17863.8 |
| [Grid Search](https://wandb.ai/xiang-pan/HOBO/runs/3vdvm6gs)| 'HNSW'     | 4  | 158            | 200 | 97.11  | 18331.748252       | -18329.638252       |



# Search Parameters Optimization



# References
<div id="refer-anchor-1"></div>
- [BOHB](https://arxiv.org/pdf/1807.01774.pdf)