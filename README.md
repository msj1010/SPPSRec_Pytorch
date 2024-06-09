
# SPPSRec-pytorch

该仓库实现了论文中的代码。

## 开始部署

### 安装以下包

这个项目需要 Python 3.8 和以下Python库:
- numpy == 1.22.0
- pandas == 1.3.5
- matplotlib == 3.5.1
- torch == 1.8.2
- fire == 0.4.0
- tqdm == 4.62.3
- loguru == 0.5.3
- dotmap == 1.3.17


```bash
pip install -r requirements.txt
```

### 数据准备

在运行模型之前，请解压数据集
```bash
unzip datasets/$DATASET.zip -d datasets/$DATASET/
```



### 如何使用
可以使用以下命令检查模型是否可以正确执行
```
PYTHONPATH=src python3 run.py --data_name $DATASET
```


## Usage
为了正确执行脚本，请将工作目录切换到 `./src`.

使用以下命令可以配置超参数并执行:

```
python3 -m main_trainer \
        --data_name ··· \
        --seed ··· \
        --epochs ··· \
        --act ··· \
        --batch_size ··· \
        --learning_rate ··· \
        --embedding_size ··· \
        --max_length ··· \
        --samples_1 ··· \
        --samples_2 ··· \
        --dropout ··· \
        --decay_rate ... \
        --gpu_id ... \
```
  


```

## 数据

### 输入数据文件
数据文件地址： 
豆瓣：[(link)](https://github.com/DeepGraphLearning/RecommenderSystems/tree/master/socialRec).
Yelp：[(link)](https://www.yelp.com/dataset).
Delicious：[(link)](https://grouplens.org/datasets/hetrec-2011/). 



