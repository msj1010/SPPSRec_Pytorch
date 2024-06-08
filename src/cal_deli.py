# -*- coding: UTF-8 -*-
import math
import numpy as np
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.cluster import KMeans

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
    #key=userID,value=关注数
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
    df_adj = pd.read_csv( './datasets/moviedata/adj.tsv', sep='\t', dtype={0: np.int32, 1: np.int32})
    user_follow_df = df_adj.groupby(by=['Follower'])['Weight'].count()
    # print(df_adj[df_adj['Followee'] == 10])
    user_follow_list = np.asarray(user_follow_df)
    max_follow_num = np.max(user_follow_list)
    min_follow_num = np.min(user_follow_list)
    mean_follow_num = np.mean(user_follow_list)
    norm_mean = ((mean_follow_num - min_follow_num) / (max_follow_num - min_follow_num))
    #key=userID,value=关注数
    user_follow_map = dict(user_follow_df)
    for k in user_follow_map.keys():
        user_follow_map[k] = ((user_follow_map[k] - min_follow_num) / (max_follow_num - min_follow_num)) ** 0.6
        if user_follow_map[k] == 0.0:
            user_follow_map[k] = norm_mean
    pickle.dump(user_follow_map,open('./src/user_follower_map.pkl','wb+'))

# def cal_item_popularity():
#
# cal_item_popularity()
def cal_user_buy():
    df = pd.read_csv('./datasets/deli.dat', sep='\t')
    user_buy_nums = dict(df.groupby(by=['userID'])['tagID'].count())
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
    pickle.dump(user_buy_nums, open('./src/user_buy_nums_deli.pkl', 'wb+'))
def cal_item_popularity():
    df = pd.read_csv('./datasets/deli.dat', sep='\t')
    item_list = set(df['bookmarkID'])
    #画图
    df.groupby(by=['userID','bookmarkID']).count()
    item_dict = dict(df.groupby(by=['bookmarkID']).count()['tagID'])
    max_follow_num = 5674
    popularity_y = np.zeros((max_follow_num+1,))
    for x in item_dict.keys():
        popularity_y[item_dict[x]] += 1
    min_num = 50
    max_num = 1200
    step = 25
    followee_x = np.arange(min_num,max_num+1,step=step)
    item_popularity = dict()
    user_buy = pickle.load(open('./src/user_buy_nums_deli.pkl', 'rb+'))
    for x in item_list:
        #sum是所有用户对项目x贡献度之和
        sum = 0.0
        buy_x_user_list = list(df[df['bookmarkID'] == x]['userID'])
        for y in buy_x_user_list:
            #用户y的购买数量
            y_buy_num = user_buy[y]
            if y_buy_num == 0:
                sum += 0
            else:
                sum += ( 1.0 / y_buy_num)
        item_popularity[x] = sum
    max_value = max(item_popularity.values())
    min_value = min(item_popularity.values())
    # normalized_dict = {key: round(((value - min_value) / (max_value - min_value)),5) for key, value in item_popularity.items()}
    #对于长尾分布使用对数归一化方法
    # normalized_dict = {key: round(math.log10(value+1),5) for key, value in normalized_dict.items()}
    #使用sigmoid归一化
    # normalized_dict = {key: round((1/(1+math.exp(-value))-0.5)*10, 5) for key, value in
    #                    item_popularity.items()}
    #使用标准归一化方法
    # data_array = [[value] for value in item_popularity.values()]
    # from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # standardized_data = scaler.fit_transform(data_array)
    # normalized_dict = {key: standardized_value[0] for key, standardized_value in
    #                      zip(item_popularity.keys(), standardized_data)}
    normalized_dict = {key: round(math.log(value*10 + 1), 5) for key, value in item_popularity.items()}
    pickle.dump(normalized_dict, open('./src/item_popularity_deli.pkl', 'wb+'))

def cluster_sequence(sequence, n_clusters):
    # 将序列转换为numpy数组
    data = np.array(sequence).reshape(-1, 1)

    # 使用K均值算法进行聚类
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)

    # 获取聚类标签
    labels = kmeans.labels_

    # 将序列按照聚类结果划分成子序列
    clustered_sequence = {}
    for i, label in enumerate(labels):
        if label not in clustered_sequence:
            clustered_sequence[label] = []
        clustered_sequence[label].append(sequence[i])

    return clustered_sequence


def popularity_session(user_id,distance_threshold = 0.1):
    df = pd.read_csv('./datasets/deli.dat', sep='\t')
    user_seq = df[df['userID'] == user_id]['bookmarkID'].values
    user_pop = np.zeros_like(user_seq,dtype=float)
    if len(user_pop) < 1:
        return user_pop,[]
    popularity_dict = pickle.load(open('./src/item_popularity_deli.pkl', 'rb+'))
    for i in range(len(user_seq)):
        user_pop[i] = popularity_dict[user_seq[i]]
    # 将序列转换为二维数组
    print(user_pop)
    X = np.array([[i, val] for i, val in enumerate(user_pop)])
    # 使用K均值聚类将序列划分为不等长的子序列
    cluster_num = 5
    if len(X) < cluster_num:
        return
    kmeans = KMeans(n_clusters=cluster_num, random_state=0, n_init=10).fit(X)
    # 获取聚类的中心点，表示子序列的划分位置
    cluster_centers = sorted(kmeans.cluster_centers_[:, 0].astype(int))
    # 根据聚类的中心点划分子序列
    sub_sequences = [
        user_pop[cluster_centers[i]:cluster_centers[i + 1]] if i < len(cluster_centers) - 1 else user_pop[cluster_centers[i]:]
        for i in range(len(cluster_centers))]
    # 输出划分后的子序列
    for i, sub_sequence in enumerate(sub_sequences):
        print(f'Subsequence {i + 1}: {sub_sequence},{len(sub_sequence)}')
    # 计算轮廓系数
    from sklearn.metrics import silhouette_score
    silhouette_avg = silhouette_score(X, kmeans.labels_)
    print(f'平均轮廊系数: {silhouette_avg}')
    from sklearn.metrics import pairwise_distances
    # 计算簇内的距离
    cluster_distances = []
    for i in range(cluster_num):  # 假设有16个簇
        cluster_points = X[kmeans.labels_ == i]
        if len(cluster_points) > 1:
            distance_matrix = pairwise_distances(cluster_points, metric='euclidean')
            cluster_distances.append(np.mean(distance_matrix))

    # 计算平均簇密度
    mean_cluster_density = np.mean(cluster_distances)
    print(f'平均聚类密度: {mean_cluster_density}')
    inertia = kmeans.inertia_
    print(f'Cluster inertia: {inertia}')
    # 计算每个样本到其所属簇中心的距离的平方和
    distances = pairwise_distances(X, kmeans.cluster_centers_, metric='euclidean')
    cohesion = np.sum(np.min(distances, axis=1))
    print(f'Cluster cohesion: {cohesion}')
    print('\n')
    #之前的聚类方法
    # from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree, fcluster
    # data_matrix = np.array([user_pop]).T
    # linkage_matrix = linkage(data_matrix, method='average')  # 这里使用层次聚类中的平均连接方法
    # clusters = fcluster(linkage_matrix, distance_threshold, criterion='distance')
    return user_pop,sub_sequences
# cal_user_follower()
#对用户聚类
def cluster_user():
    df = pd.read_csv('./datasets/moviedata/train.tsv', sep='\t')
    user_buy_nums = df.groupby(by=['userID'])['userID'].count()
    cluster = {}
    user_id_list = np.array(list(set(df['userID']))).reshape(-1,1)
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
df = pd.read_csv('./datasets/deli.dat', sep='\t')
user_ids = list(set(df['userID'].values))
print(user_ids)
for i in user_ids:
    popularity_session(user_id=i,distance_threshold=0.75)



