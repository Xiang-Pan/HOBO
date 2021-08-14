<!--
 * @Author: Xiang Pan
 * @Date: 2021-08-13 18:00:59
 * @LastEditTime: 2021-08-13 19:26:35
 * @LastEditors: Xiang Pan
 * @Description: 
 * @FilePath: /HOBO/REPORT.md
 * xiangpan@nyu.edu
-->

**Threshold 95**

Note: Grid Search is done using zilliz server and BO is based on my local machine.
# Server Hardwareware Information

# Hardwareware Information
CPU: Intel Core i9-10900F @ 20x 5.2GHz  
GPU: NVIDIA GeForce RTX 3090  
RAM: 31908MiB  
# Index Type Optimization

| Method      | index_type | M  | efConstruction | ef  | recall | query_per_sec      | loss                |
|-------------|------------|----|----------------|-----|--------|--------------------|---------------------|
| BO          | 'HNSW'     | 9 | 255             | 116 | 98.71 | 20035.845992165854 | -20032.135992165855 |


|             | index_type | nlist | M   | nprobe | recall | query_per_sec    | loss              |
|-------------|------------|-------|-----|--------|--------|------------------|-------------------|
| Grid Search | 'IVF_PQ'   | 301   | 8   | 30     | 96.94  | 18080.455212     | -18078.515212     |



# Index Parameters Optimization


## IVF_FLAT

|             | index_type | nlist | nprobe | recall | query_per_sec      | loss                |
|-------------|------------|-------|--------|--------|--------------------|---------------------|
| BO          | 'IVF_FLAT' | 13305 | 31     | 97.22  | 17261.931023129477 | -17259.711023129476 |
| Grid Search | 'IVF_FLAT' | 14601 | 101    | 100.0  | 14402.032758       | -14397.032758       |

## IVF_SQ8
|             | index_type | nlist | nprobe | recall | query_per_sec     | loss               |
|-------------|------------|-------|--------|--------|-------------------|--------------------|
| BO          | 'IVF_SQ8'  | 883   | 31     | 97.07  | 17455.17499687877 | -17453.10499687877 |
| Grid Search | 'IVF_SQ8'  | 5401  | 101    | 99.49  | 13080.62997       | -13076.13997       |

## IVF_PQ
|             | index_type | nlist | M   | nprobe | recall | query_per_sec    | loss              |
|-------------|------------|-------|-----|--------|--------|------------------|-------------------|
| BO          | 'IVF_PQ'     | 17    | 445 | 106    | 99.55  | 20555.2756677285 | -20550.7256677285 |
| Grid Search | 'IVF_PQ'   | 301   | 8   | 30     | 96.94  | 18080.455212     | -18078.515212     |


## HNSW
| Method      | index_type | M  | efConstruction | ef  | recall | query_per_sec      | loss                |
|-------------|------------|----|----------------|-----|--------|--------------------|---------------------|
| BO          | 'HNSW'     | 15 | 16             | 157 | 97.52  | 19157.321640632137 | -19154.801640632137 |
| Grid Search | 'HNSW'     | 4  | 158            | 200 | 97.11  | 18331.748252       | -18329.638252       |



# Search Parameters Optimization