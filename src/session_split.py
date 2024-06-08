# -*- coding: UTF-8 -*-
import math
import numpy as np
import json
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
# plt.style.use("ggplot")
#计算用户U被几个用户关注
def cal_user_interval():
    user_id = 1016
    count = 0
    df = pd.read_csv( './datasets/deli.dat', sep='\t')
    timestamp = df[df['userID'] == user_id]['timestamp']
    timestamp = timestamp.sort_values(ascending=True)
    timestamp = np.asarray(timestamp,dtype=float)
    # timestamp = np.sort(timestamp)
    inteval_arr = []
    for i in range(timestamp.shape[0]-1):
        inteval = (timestamp[i+1] - timestamp[i])/100/60
        inteval = round(inteval,2)
        if inteval >= 271:
            print('划分会话')
            count += 1
        inteval_arr.append(inteval)
    print(inteval_arr)
    print('shape',timestamp.shape[0])
    print('count',count)
    interval_avg = np.average(inteval_arr)
    inteval_arr_med = []
    for i in range(len(inteval_arr)):
        if inteval_arr[i] != 0.0:
            inteval_arr_med.append(inteval_arr[i])
    interval_med = np.median(inteval_arr_med)
    print(interval_avg,interval_med)

#使用聚类划分会话
def cluster_session():
    user_id = 1016
    count = 0
    df = pd.read_csv( './datasets/deli.dat', sep='\t')
    seq = df[df['userID'] == user_id]['bookmarkID']
    print(seq)
    import numpy as np
    from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree
    import matplotlib.pyplot as plt
    np_seq = np.array([seq.values]).T
    print(np_seq)
    # 计算相似性矩阵
    linkage_matrix = linkage(np_seq, method='average')
    # 根据截断点生成簇
    cut_threshold = 0.5  # 根据需要调整
    clusters = cut_tree(linkage_matrix, height=cut_threshold)

    # 可视化层次聚类树
    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sequence Index')
    plt.ylabel('Distance')
    plt.show()
    # clusters包含每个序列所属的簇标签
    print("Clusters:", clusters.flatten())

# cal_user_interval()
from datetime import datetime
def calculate_date_difference(date_str1, date_str2):
    # 将日期字符串转换为datetime对象
    date1 = datetime.strptime(date_str1, '%Y-%m-%d %H:%M:%S')
    date2 = datetime.strptime(date_str2, '%Y-%m-%d %H:%M:%S')
    # 计算日期差异
    date_difference = date2 - date1
    # 提取差异的天数部分
    days_difference = date_difference.days
    return days_difference
def yelp():

    data_file = open("./datasets/yelp_split.dat",'rb')
    user_id = '8g_iMtfSiwikVnbP2etR0A'
    checkin_df = pickle.load(data_file)
    checkin_df = checkin_df[checkin_df['user_id'] == user_id]['date']
    checkin_df = checkin_df.sort_values(ascending=True)
    checkin_np = np.asarray(checkin_df)
    inteval_arr = []
    for i in range(checkin_np.shape[0] - 1):
        inteval = calculate_date_difference(checkin_np[i],checkin_np[i+1])
        inteval_arr.append(inteval)
    print(inteval_arr)
    # print(checkin_df)
    inteval_arr = np.asarray(inteval_arr)
    inteval_ave = np.average(inteval_arr)
    inteval_med = np.median(inteval_arr)
    print('平均天数',inteval_ave,'中位数',inteval_med)
    data_file.close()


def yelp2():

    data_file = open("./datasets/yelp.json",encoding='utf-8',errors='ignore')
    data = []
    for line in data_file:
        data.append(json.loads(line))
    checkin_df = pd.DataFrame(data)[['review_id','user_id','date']]
    checkin_df = checkin_df.head(1000000)
    print(checkin_df)
    split_dat = open('./datasets/yelp_split.dat','wb+')
    pickle.dump(checkin_df,split_dat)
    data_file.close()

cluster_session()


