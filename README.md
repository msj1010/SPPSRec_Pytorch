
# DGRec-pytorch

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

### Data Preparation

在运行模型之前，请解压数据集
```bash
unzip datasets/$DATASET.zip -d datasets/$DATASET/
```



### How To Run
可以使用以下命令检查模型是否可以正确执行
```
PYTHONPATH=src python3 run.py --data_name $DATASET
```


## Usage
To use those scripts properly, move your working directory to `./src`.

You can tune the hyperparameters and run this project to simply type the following in your terminal:

```
python3 -m main_trainer \
        --model ··· \
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

## Data

### Input Data Files
The datasets are from the original repository [(link)](https://github.com/DeepGraphLearning/RecommenderSystems/tree/master/socialRec). The format of each file is as follows:

* `train.tsv`: includes historical behaviors of users. The data in the file is loaded into pandas.Dataframe with five fields such as (SessionId UserId ItemId Timestamps TimeId)
* `valid.tsv`: the same format as `train.tsv`, used for tuning hyperparameters.
* `test.tsv`: the same format as `train.tsv`, used for testing model.
* `adj.tsv`: an edge list representing relationships between users, which is also organized by pandas.Dataframe in two fields (FromId, ToId).
* `latest_session.tsv`: serves as 'reference' to target user. This file records all users available session at each time slot. For example, at time slot t, it stores user u's t-1 th session.
* `user_id_map.tsv`: maps original string user name to integer id.
* `item_id_map.tsv`: maps original string item name to integer id.

### Data Statistics
The statistics of `bookdata`, `musicdata`, and `moviedata` from the Douban domain are summarized as follows:

|Dataset|#user|#item|#event|
|------|---|---|---|
|`bookdata`|46,548|212,995|1,908,081|
|`musicdata`|39,742|164,223|1,792,501|
|`moviedata`|94,890|81,906|11,742,260|

