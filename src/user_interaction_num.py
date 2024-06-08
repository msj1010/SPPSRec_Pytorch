import pandas as pd


def cal_num(user_id=0):
    df = pd.read_csv('./datasets/moviedata/train.tsv', sep='\t')
    # user_id = 0
    user_seq = df[df['UserId'] == user_id]['ItemId'].values
    print('userid:{},num:{}'.format(user_id,len(user_seq)))

for i in range(2000):
    cal_num(i)