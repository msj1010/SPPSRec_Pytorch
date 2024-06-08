# -*- coding: UTF-8 -*-
import math
import numpy as np
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
# plt.style.use("ggplot")
#计算用户U被几个用户关注
plt.rcParams['font.sans-serif']=['SimHei']  #解决中文显示乱码问题
def cal_user_followee():
    df_adj = pd.read_csv( './datasets/moviedata/adj.tsv', sep='\t', dtype={0: np.int32, 1: np.int32})
    user_follow_df = df_adj.groupby(by=['Followee'])['Weight'].count()
    # print(df_adj[df_adj['Followee'] == 10])
    user_follow_list = np.asarray(user_follow_df)

    max_follow_num = np.max(user_follow_list)
    min_follow_num = np.min(user_follow_list)
    followee_y = np.zeros((max_follow_num+1,))
    for x in user_follow_list:
        followee_y[x] += 1
    max_num = 50
    followee_x = np.linspace(1,max_num,max_num)
    plt.plot(followee_x,followee_y[1:max_num+1])
    plt.xlabel('粉丝数')
    plt.ylabel('用户数')
    plt.title('用户粉丝数的分布')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.legend()
    plt.show()
    return
    mean_follow_num = np.mean(user_follow_list)
    norm_mean = ((mean_follow_num - min_follow_num) / (max_follow_num - min_follow_num)) ** 0.15
    #key=userid,value=关注数
    user_follow_map = dict(user_follow_df)
    for k in user_follow_map.keys():
        user_follow_map[k] = ((user_follow_map[k] - min_follow_num) / (max_follow_num - min_follow_num)) ** 0.15
        # user_follow_map[k] = math.log10(user_follow_map[k])
        # user_follow_map[k] = (math.sin(user_follow_map[k]) + 1) / 2
        if user_follow_map[k] == 0.0:
            user_follow_map[k] = norm_mean
    pickle.dump(user_follow_map,open('./src/user_followee_map.pkl','wb+'))

#计算用户U关注了几个用户
def cal_user_follower():
    df_adj = pd.read_csv( './datasets/bookdata/adj.tsv', sep='\t', dtype={0: np.int32, 1: np.int32})
    user_follow_df = df_adj.groupby(by=['Follower'])['Weight'].count()
    # print(df_adj[df_adj['Followee'] == 10])
    user_follow_list = np.asarray(user_follow_df)
    max_follow_num = np.max(user_follow_list)
    min_follow_num = np.min(user_follow_list)
    mean_follow_num = np.mean(user_follow_list)
    norm_mean = ((mean_follow_num - min_follow_num) / (max_follow_num - min_follow_num))
    #key=userid,value=关注数
    user_follow_map = dict(user_follow_df)
    for k in user_follow_map.keys():
        user_follow_map[k] = ((user_follow_map[k] - min_follow_num) / (max_follow_num - min_follow_num)) ** 0.6
        if user_follow_map[k] == 0.0:
            user_follow_map[k] = norm_mean
    pickle.dump(user_follow_map,open('./src/user_book_follower_map.pkl','wb+'))

# def cal_item_popularity():
#
# cal_item_popularity()
def cal_user_buy():
    df = pd.read_csv('./datasets/moviedata/train.tsv', sep='\t')
    user_buy_nums = dict(df.groupby(by=['UserId'])['Rating'].count())
    # user_buy_list = np.asarray(list(user_buy_nums.values()))
    # max_follow_num = np.max(user_buy_list)
    # min_follow_num = np.min(user_buy_list)
    # mean_follow_num = np.mean(user_buy_list)
    # var_buy_num = np.var(user_buy_list)
    # norm_mean = ((mean_follow_num - min_follow_num) / (max_follow_num - min_follow_num))
    # for k in user_buy_nums.keys():
    #     user_buy_nums[k] = math.fabs(user_buy_nums[k] - mean_follow_num) / var_buy_num
    #     # user_buy_nums[k] = (math.sin(user_buy_nums[k])+1) / 2
    #     user_buy_nums[k] = math.log10(user_buy_nums[k])
    #     if user_buy_nums[k] == 0.0:
    #         user_buy_nums[k] = math.log10(norm_mean)
    pickle.dump(user_buy_nums, open('./src/user_buy_nums.pkl', 'wb+'))
def cal_item_popularity():
    df = pd.read_csv('./datasets/bookdata/train.tsv', sep='\t')
    item_list = set(df['ItemId'])
    #画图
    df.groupby(by=['UserId','ItemId']).count()
    item_dict = dict(df.groupby(by=['ItemId']).count()['Rating'])
    max_follow_num = 5674
    popularity_y = np.zeros((max_follow_num+1,))
    for x in item_dict.keys():
        popularity_y[item_dict[x]] += 1
    min_num = 50
    max_num = 1200
    step = 25
    followee_x = np.arange(min_num,max_num+1,step=step)

    plt.plot(followee_x, popularity_y[min_num:max_num+1:step])
    plt.fill_between(followee_x,0, popularity_y[min_num:max_num + 1:step],facecolor='green', alpha=0.3)
    plt.xlabel('购买次数')
    plt.ylabel('项目数')
    plt.title('项目购买数的分布')
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.legend()
    plt.show()
    item_popularity = dict()
    user_follow_map = pickle.load(open('./src/user_follow_map.pkl', 'rb+'))
    user_buy = pickle.load(open('./src/user_buy_nums.pkl', 'rb+'))
    for x in item_list:
        #sum是所有用户对项目x贡献度之和
        sum = 0.0
        buy_x_user_list = list(df[df['ItemId'] == x]['UserId'])
        for y in buy_x_user_list:
            #用户y的购买数量
            y_buy_num = user_buy[y]
            #用户y的信任值
            y_trust = user_follow_map.get(y,1)
            if y_buy_num == 0:
                sum += 0
            else:
                sum += ( 1.0 / y_buy_num)
        # print('sum:',sum)
        item_popularity[x] = sum
    max_value = max(item_popularity.values())
    min_value = min(item_popularity.values())
    normalized_dict = {key: round(((value - min_value) / (max_value - min_value)) ** 0.2,3) for key, value in item_popularity.items()}
    pickle.dump(normalized_dict, open('./src/item_book_popularity.pkl', 'wb+'))

def popularity_session(user_id=0):
    df = pd.read_csv('./datasets/moviedata/train.tsv', sep='\t')
    # user_id = 0
    user_seq = df[df['UserId'] == user_id]['ItemId'].values
    print('user_seq',user_seq)
    user_pop = np.zeros_like(user_seq,dtype=float)
    # print('seq:{},pop:{}\n'.format(len(user_seq),len(user_pop)))
    popularity_dict = pickle.load(open('./src/item_popularity.pkl', 'rb+'))
    # print('pop_1',popularity_dict[1])
    # print('商品流行度',popularity_dict)
    for i in range(len(user_seq)):
        user_pop[i] = popularity_dict[user_seq[i]]
    # print(user_pop)
    from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree, fcluster
    import matplotlib.pyplot as plt
    data_matrix = np.array([user_pop]).T
    linkage_matrix = linkage(data_matrix, method='average')  # 这里使用层次聚类中的平均连接方法

    # 根据截断点生成簇
    # cut_threshold = 0.5  # 根据需要调整
    # clusters = cut_tree(linkage_matrix, height=cut_threshold)
    # 根据距离阈值截断树状图，形成簇
    distance_threshold = 0.095  # 根据需要调整
    clusters = fcluster(linkage_matrix, distance_threshold, criterion='distance')

    # 输出簇标签
    # print("Clusters:", clusters)
    # 可视化簇划分（可选）
    plt.figure(figsize=(10, 6))
    plt.plot(user_pop, label='Normalized Sequence', marker='o')
    for cluster_label in np.unique(clusters):
        indices = np.where(clusters == cluster_label)[0]
        plt.plot(indices, [user_pop[i] for i in indices], label=f'Cluster {cluster_label}', linestyle='--',
                 marker='o')
    plt.title('Clustering Result')
    plt.xlabel('Sequence Index')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.show()
    # clusters包含每个序列所属的簇标签
    # print("Clusters:", clusters.flatten())
    return user_pop,clusters.flatten().tolist()
cal_user_follower()
cal_item_popularity()
#对用户聚类
def cluster_user():
    df = pd.read_csv('./datasets/moviedata/train.tsv', sep='\t')
    user_buy_nums = df.groupby(by=['UserId'])['UserId'].count()
    cluster = {}
    user_id_list = np.array(list(set(df['UserId']))).reshape(-1,1)
    #读入用户购买数量,放入第1列
    user_buy_nums = pickle.load(open('./src/user_buy_nums.pkl', 'rb+'))
    # 读入用户主动关注人数,放入第2列
    user_follower = pickle.load(open('./src/user_follower_map.pkl', 'rb+'))
    # 读入用户被关注人数,放入第3列
    user_followee = pickle.load(open('./src/user_followee_map.pkl', 'rb+'))

    for x in user_buy_nums.keys():
        cluster[x] = (user_buy_nums.get(x, 0), user_follower.get(x, 0), user_followee.get(x, 0))
    max_user_id = max(list(cluster.keys()))
    user_cluster = np.zeros((max_user_id+1,3))
    for x in cluster.keys():
        if cluster[x][0] == 0 or cluster[x][1] == 0 or cluster[x][2] == 0 :
            continue
        user_cluster[x] = [cluster[x][0],cluster[x][1],cluster[x][2]]
    pickle.dump(user_cluster,open('./src/user_cluster.pkl','wb+'))
    from sklearn.cluster import MiniBatchKMeans
    import matplotlib.pyplot as plt
    # 假如我要构造一个聚类数为3的聚类器
    db = MiniBatchKMeans(n_clusters=4,batch_size=300, random_state=218).fit(user_cluster)
    # user_cluster.
    plt.scatter(user_cluster[:, 0], user_cluster[:, 2], c=db.labels_)
    plt.ylim(0.25,3)
    plt.show()

def split_sequence_by_majority_interval(sequence, threshold_percentage=0.2,min_length=10):
    result_sequences = []
    current_sequence = [sequence[0]]
    current_count = 1

    for i in range(1, len(sequence)):
        if sequence[i] == sequence[i - 1]:
            current_sequence.append(sequence[i])
            current_count += 1
        else:
            # Check if the current count exceeds the threshold percentage
            if current_count / len(current_sequence) >= threshold_percentage and len(current_sequence) >= min_length:
                result_sequences.append(current_sequence)
                current_sequence = [sequence[i]]
                current_count = 1
                continue
            else:
                current_sequence.append(sequence[i])
            if len(current_sequence) >= min_length:
                result_sequences.append(current_sequence)
                current_sequence = [sequence[i]]
                current_count = 1

    # Check for the last sequence
    result_sequences.append(current_sequence)
    return result_sequences

