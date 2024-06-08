# Template code is provided at the
# https://github.com/DeepGraphLearning/RecommenderSystems/blob/master/socialRec/dgrec/utils.py

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import pickle

class MyDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path

    def load_adj(self, data_path):#构建用户之间关注与被关注的关系图
        df_adj = pd.read_csv(data_path + '/adj.tsv', sep='\t', dtype={0: np.int32, 1: np.int32})
        pickle.dump(df_adj,open('./df_adj.pkl','wb+'))
        #构建用户-被关注数的dict
        return df_adj

    def load_latest_session(self, data_path):
        ret = []
        for line in open(data_path + '/latest_sessions.txt'):
            chunks = line.strip().split(',')
            ret.append(chunks)
        pickle.dump(ret,open('./ret.pkl','wb+'))
        return ret

    def load_map(self, data_path, name='user'):
        if name == 'user':
            file_path = data_path + '/user_id_map.tsv'
        elif name == 'item':
            file_path = data_path + '/item_id_map.tsv'
        else:
            raise NotImplementedError
        id_map = {}
        for line in open(file_path):
            k, v = line.strip().split('\t')
            id_map[k] = str(v)
        pickle.dump(id_map,open('./id_map.pkl','wb+'))
        return id_map

    def load_data(self):
        try:
            with open('./all_data.pkl','rb+') as f:
                all_data = pickle.load(f)
                return all_data
        except:
            pass
        adj = self.load_adj(self.data_path)
        latest_sessions = self.load_latest_session(self.data_path)
        user_id_map = self.load_map(self.data_path, 'user')
        item_id_map = self.load_map(self.data_path, 'item')
        train = pd.read_csv(self.data_path + '/train.tsv', sep='\t', dtype={0: np.int32, 1: np.int32, 3: np.float32})
        valid = pd.read_csv(self.data_path + '/valid.tsv', sep='\t', dtype={0: np.int32, 1: np.int32, 3: np.float32})
        test = pd.read_csv(self.data_path + '/test.tsv', sep='\t', dtype={0: np.int32, 1: np.int32, 3: np.float32})
        all_data = [adj, latest_sessions, user_id_map, item_id_map, train, valid, test]
        pickle.dump(all_data,open('./all_data.pkl','wb+'))
        return all_data
