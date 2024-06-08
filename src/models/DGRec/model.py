#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
from loguru import logger

class FrequencyLayer(nn.Module):
    def __init__(self, args):
        super(FrequencyLayer, self).__init__()
        self.out_dropout = nn.Dropout(0.5)
        self.LayerNorm = LayerNorm(50, eps=1e-12)
        self.c = args
        self.beta = nn.Parameter(torch.randn(1, 1, 50))

    def forward(self, input_tensor):
        # [batch, seq_len, hidden]
        batch, seq_len, hidden = input_tensor.shape
        x = torch.fft.rfft(input_tensor, dim=1, norm='ortho')

        low_pass = x[:]
        low_pass[:, self.c:, :] = 0
        low_pass = torch.fft.irfft(low_pass, n=seq_len, dim=1, norm='ortho')
        high_pass = input_tensor - low_pass
        sequence_emb_fft = low_pass + (self.beta**2) * high_pass

        hidden_states = self.out_dropout(sequence_emb_fft)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class GAT(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0., bias=False, act=nn.ReLU(), **kwargs):
        super().__init__()

        self.act = act

        self.feat_drop = nn.Dropout(dropout) if dropout > 0 else None
        self.fc = nn.Linear(input_dim, output_dim, bias=True)
        self.fc.weight = torch.nn.init.xavier_uniform_(self.fc.weight)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inputs):
        self_vecs, neigh_vecs = inputs

        if self.feat_drop is not None:
            self_vecs = self.feat_drop(self_vecs)
            neigh_vecs = self.feat_drop(neigh_vecs)

        # Reshape from [batch_size, depth] to [batch_size, 1, depth] for matmul.
        self_vecs = torch.unsqueeze(self_vecs, 1)  # [batch, 1, embedding_size] 目标用户的节点表示
        neigh_self_vecs = torch.cat((neigh_vecs, self_vecs), dim=1)  # [batch, sample, embedding] 目标用户邻居的节点表示
        #第k个朋友对用户u的影响权重分
        score = self.softmax(torch.matmul(self_vecs, torch.transpose(neigh_self_vecs, 1, 2)))
        #用户Fu与其朋友兴趣的混合表示
        # alignment(score) shape is [batch_size, 1, depth]
        context = torch.squeeze(torch.matmul(score, neigh_self_vecs), dim=1)

        # [nodes] x [out_dim]
        output = self.act(self.fc(context))

        return output


#论文创新，加入CNN，如调不通就删除
class cnnBlock(nn.Module):

    def __init__(self, embed_dim=50, hidden_dim=100, num_filters=10, kernel_size=3, output_dim=10):
        super(cnnBlock, self).__init__()
        # 将嵌入维度扩展到hidden_dim
        self.embed_to_hidden = nn.Linear(embed_dim, hidden_dim)
        # 一维卷积层
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=kernel_size)
        # 池化层
        self.pool = nn.MaxPool1d(kernel_size=2)
        # 全连接层
        self.fc = nn.Linear(num_filters * ((hidden_dim - kernel_size + 1) // 2), output_dim)

    def forward(self, x):
        # x的维度是(450, 20, 50)
        # 首先将每个商品的嵌入通过线性层扩展到hidden_dim
        x = self.embed_to_hidden(x.view(-1, 50))  # 转换维度到(20 * 450, 50)
        x = x.view(-1, 1, x.size(1))  # 转换维度到(20, 1, hidden_dim)，以适应Conv1d
        # 应用一维卷积
        x = self.conv1d(x)
        # 应用池化
        x = self.pool(x)
        # 展平以适配全连接层
        x = x.view(-1, x.size(1) * x.size(2))
        # 全连接层
        x = self.fc(x)
        return x

class GGNN(nn.Module):
    # Gated Graph Neural Network
    def __init__(self, hidden_size, step=1):
        super(GGNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size

        self.fc_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.fc_rzh_input = nn.Linear(2 * self.hidden_size, 3 * self.hidden_size, bias=True)
        self.fc_rz_old = nn.Linear(1 * self.hidden_size, 2 * self.hidden_size, bias=True)
        self.fc_h_old = nn.Linear(1 * self.hidden_size, 1 * self.hidden_size, bias=True)

    def aggregate(self, A, emb_items):
        h_input_in = self.fc_edge_in(torch.matmul(A[:, :, :A.shape[1]], emb_items))
        h_input_out = self.fc_edge_out(torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], emb_items))
        h_inputs = torch.cat([h_input_in, h_input_out], 2)

        r_input, z_input, h_input = self.fc_rzh_input(h_inputs).chunk(chunks=3, dim=2)
        r_old, z_old = self.fc_rz_old(emb_items).chunk(chunks=2, dim=2)

        reset = torch.sigmoid(r_old + r_input)
        update = torch.sigmoid(z_old + z_input)
        h = torch.tanh(h_input + self.fc_h_old(reset * emb_items))
        return (1 - update) * emb_items + update * h

    def forward(self, A, h_item):
        for i in range(self.step):
            h_item = self.aggregate(A, h_item)
        return h_item


class DGRec(torch.nn.Module):
    def __init__(
            self,
            hyper_param,
            num_layers,
    ):
        super(DGRec, self).__init__()
        self.act = hyper_param['act']
        self.batch_size = hyper_param['batch_size']
        self.num_users = hyper_param['num_users']
        self.num_items = hyper_param['num_items']
        self.embedding_size = hyper_param['embedding_size']
        self.max_length = hyper_param['max_length']
        self.samples_1 = hyper_param['samples_1']
        self.samples_2 = hyper_param['samples_2']
        # self.dropout = hyper_param['dropout']
        self.dropout = 0
        self.num_layers = num_layers
        self.alpha = 0.9
        if self.act == 'relu':
            self.act = nn.ReLU()
        elif self.act == 'elu':
            self.act = nn.ELU()


        self.user_embedding = nn.Embedding(self.num_users,
                                           self.embedding_size)  # (num_users=26511, embedding_dim=100)
        #流行度嵌入层,流行度最大为1247,记为1300
        self.pop_embedding = nn.Embedding(1300,
                                           self.embedding_size)
        self.followee_embedding = nn.Embedding(120,
                                          self.embedding_size)
        self.GGNN = GGNN(self.embedding_size,1)
        # 论文创新加入个人短期偏好卷积层
        self.conv_v = nn.Conv2d(1, 1, (5,1),stride=1,padding=(2,0))
        #用于处理朋友的长期兴趣
        self.friends_long_tern = nn.Linear(self.embedding_size,self.embedding_size)
        self.fri_influence = nn.Linear(self.max_length,1)
        # self.dynastic_conv = DynamicConv(1,1,1,padding=(1,1))
        self.item_embedding = nn.Embedding(self.num_items,
                                           self.embedding_size,
                                           padding_idx=0)  # (num_items=12592, embedding_dim=100)
        self.item_indices = nn.Parameter(torch.arange(0, self.num_items, dtype=torch.long),
                                         requires_grad=False)
        self.attn = nn.MultiheadAttention(self.embedding_size,num_heads=1,dropout=self.dropout)
        self.feat_drop = nn.Dropout(self.dropout) if self.dropout > 0 else None
        input_dim = self.embedding_size
        self.conv = nn.Conv1d(in_channels=20, out_channels=1, kernel_size=1)
        # making user embedding
        #创新点：使用双向LSTM机制
        self.lstm = nn.LSTM(self.embedding_size, self.embedding_size, batch_first=True,bidirectional=True)
        # self.sru = SRU(self.embedding_size,self.embedding_size)
        self.friends_lstm = nn.LSTM(self.embedding_size, self.embedding_size, batch_first=True)
        #论文创新，卷积层
        block_dim = [100,100,100]
        self.block_num = 2
        # block_dim.insert(0, 2 * self.embedding_size)
        self.cnnBlock = cnnBlock()
        # self.avgpool_1 = nn.AdaptiveAvgPool2d((1, self.samples_2 * self.samples_1 * self.batch_size))
        # self.avgpool_2 = nn.AdaptiveAvgPool2d((1,  self.samples_1 * self.batch_size))
        # self.avgpool_arr = [self.avgpool_1,self.avgpool_2]
        # 傅里叶变换
        self.filter_layer = FrequencyLayer(2)
        # combine friend's long and short-term interest
        self.W1 = nn.Linear(2 * self.embedding_size, self.embedding_size, bias=False)
        self.W1.weight = torch.nn.init.xavier_uniform_(self.W1.weight)

        # combine user interest and social influence
        self.W2 = nn.Linear(input_dim + self.embedding_size, self.embedding_size, bias=False)
        self.W2.weight = torch.nn.init.xavier_uniform_(self.W2.weight)

        # making GAT layers
        self.layers = nn.ModuleList()
        for layer in range(num_layers):
            aggregator = GAT(input_dim, input_dim, act=self.act, dropout=self.dropout)
            self.layers.append(aggregator)

    # get target user's interest

    def get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


    #增加流行度参数
    def individual_interest(self, input_session,input_pop):
        input = input_session[0].long()  # input.shape : [max_length]
        #个人的某个会话序列emb表示
        emb_seqs = self.item_embedding(input)
        # emb_seqs.shape : [max_length, embedding_dim]
        #与个人会话序列对应的流行度序列emb表示
        emb_pops = self.pop_embedding(input_pop)
        emb_seqs = torch.unsqueeze(emb_seqs, 0)

        if self.feat_drop is not None:
            emb_seqs = self.feat_drop(emb_seqs)

        for batch in range(self.batch_size - 1):
            input = input_session[batch + 1].long()
            emb_seq = self.item_embedding(input)
            emb_seq = torch.unsqueeze(emb_seq, 0)
            emb_seqs = torch.cat((emb_seqs, emb_seq), 0)
        # nn.MultiheadAttention()
        #论文创新,将流行度序列融合进用户交互序列
        emb_seqs += emb_pops
        emb_seqs = torch.squeeze(emb_seqs,dim=1)
        # 做傅里叶变换
        dsp = self.filter_layer(emb_seqs)
        # 输出后 [b*len*dim]
        attn, _ = self.attn(emb_seqs, emb_seqs, emb_seqs)
        hidden_states = self.alpha * dsp + (1 - self.alpha) * attn


        return hidden_states

    # get friends' interest
    def friends_interest(self, support_nodes_layer1, support_nodes_layer2, support_sessions_layer1,
                         support_sessions_layer2,layer1_followee,layer2_followee,session1_pop,session2_pop,mask_y):
        ''' long term '''
        long_term = []
        #其中layer1是用户的朋友，layer2是朋友的朋友
        support_nodes_layer = [support_nodes_layer1, support_nodes_layer2]
        session1_pop_emb = self.pop_embedding(session1_pop)
        session2_pop_emb = self.pop_embedding(session2_pop)
        pop_emb_arr = [session1_pop_emb,session2_pop_emb]
        pop_out_arr = []
        #将用户被关注数融入模型
        layer1_followee_emb = self.followee_embedding(layer1_followee)
        layer2_followee_emb = self.followee_embedding(layer2_followee)
        # attn = nn.MultiheadAttention(embed_dim=100, dropout=0.1, num_heads=1)
        # weight, _ = attn(session1_pop_emb, session1_pop_emb, session1_pop_emb)
        user_followee_layer = [layer1_followee_emb,layer2_followee_emb]
        support_sessions_layer = [support_sessions_layer1, support_sessions_layer2]
        count = 0
        for layer in support_sessions_layer:
            #用户交互行为传入线性层
            mask_batch = self.get_attention_mask(layer)
            fri_long = self.item_embedding(layer)
            # long_term_ = self.friends_long_tern(long_term)
            #用户社交影响
            # user_influence = pop_emb_arr[count]
            user_influence = user_followee_layer[count] #450 * 50
            #450*20*50 -> 450 * 50 * 1
            alpha_att = torch.sigmoid(self.fri_influence(fri_long.transpose(1,2))).transpose(1,2)
            seq_emb = torch.sum(alpha_att * fri_long, 1)
            long_term.append(seq_emb)
            count += 1
        #提取朋友的长期偏好，原来的想法是使用朋友的个人嵌入，现在更换掉原来的方法

        # for layer in support_nodes_layer:
        #     long_input = layer.long()
        #     long_term1_2 = self.user_embedding(long_input)
        #     #将朋友关注的其他用户嵌入与该用户的粉丝数进行融合
        #     long_term1_2 += user_followee_layer[count]
        #     long_term.append(long_term1_2)
        #     #long term 0 [25000 * 100] 1 [2500 100]
        #     count += 1
            # long_term[0].shape : [sample1 * sample2, embedding_dim]
            # long_term[1].shape : [sample2, embedding_dim]
        # long_term[1],_ = self.attn(long_term[1],long_term[1],long_term[1])
        ''' short term '''
        short_term = []
        #短期兴趣使用的是朋友最新的会话

        sample1_2 = [self.samples_1 * self.samples_2, self.samples_2]
        index = 0
        for layer, sample in zip(support_sessions_layer, sample1_2):
            short_arange = torch.arange(self.batch_size * sample, dtype=torch.long)
            short_input = layer[short_arange].long()
            #[25000 20 100]
            friend_emb_seqs = self.item_embedding(short_input)

            if self.feat_drop is not None:
                friend_emb_seqs = self.feat_drop(friend_emb_seqs)
            # friend_emb_seqs,_ = self.attn(friend_emb_seqs,friend_emb_seqs,friend_emb_seqs)
            # friend_emb_seqs = self.conv_v(friend_emb_seqs)
            # short_term1_2 = self.conv_v_lstm(friend_emb_seqs,friend_emb_seqs)
            
            # _, (_, short_term1_2) = self.lstm(friend_emb_seqs)
            #1 * 50 *20 * 450，pop_out与out的shape相同
            out = torch.unsqueeze(friend_emb_seqs,dim=0).permute(0,3,2,1)
            pop_out = torch.unsqueeze(pop_emb_arr[index],dim=0).permute(0,3,2,1)
            # for i in range(self.block_num):
            #     out = self.cnnBlock[i](out)
            #     pop_out = self.cnnBlock[i](pop_out)
            # #[1 * emb * max_length * 2500]
            # out = self.avgpool_arr[index](out)
            # pop_out = self.avgpool_arr[index](pop_out)
            index += 1
            # out.reshape_(friend_emb_seqs.shape[0], 100)
            short_term1_2 = out.permute(0,3,2,1).squeeze(0).squeeze(2).squeeze(1)
            pop_out = pop_out.permute(0,3,2,1).squeeze(0).squeeze(2).squeeze(1)
            # short_term1_2 = torch.squeeze(short_term1_2)
            # short_term1_2 = self.attn(short_term1_2,short_term1_2,short_term1_2,)
            # (_,short_term1_2) = self.lstm(short_term1_2)
            # short_term1_2 = short_term1_2.reshape(short_term1_2.shape[0],short_term1_2.shape[1] * short_term1_2.shape[2])
            short_term.append(short_term1_2)
            pop_out_arr.append(pop_out)
            # short_term[0].shape : [batch_size * sample1 * sample2, embedding_dim]
            # short_term[1].shape : [batch_size * sample2, embedding_dim]
        short_term[1],_ = self.attn(short_term[1]+pop_out_arr[1],short_term[1]+pop_out_arr[1],short_term[1]+pop_out_arr[1])

        for i in range(0,2):
            short_term[i] = F.relu(self.conv(short_term[i])).squeeze(1)
        ''' long-short term'''
        # short_term[0].reshape(short_term[0].shape[0],short_term[0].shape[1] * )
        long_short_term1 = torch.cat((long_term[0], short_term[0]), dim=1)
        long_short_term2 = torch.cat((long_term[1], short_term[1]), dim=1)

        # long_short_term1.shape : [batch_size * sample1 * sample2, embedding_dim + embedding_dim]
        # long_short_term2.shape : [batch_size * sample2, embedding_dim + embedding_dim]

        if self.feat_drop is not None:
            long_short_term1 = self.feat_drop(long_short_term1)
            long_short_term2 = self.feat_drop(long_short_term2)

        long_short_term1 = torch.relu(self.W1(long_short_term1))
        long_short_term2 = torch.relu(self.W1(long_short_term2))
        # long_short_term1.shape : [batch_size * sample1 * sample2, embedding_dim]
        # long_short_term2.shape : [batch_size * sample2, embedding_dim]

        long_short_term = [long_short_term2, long_short_term1]

        return long_short_term

    # get user's interest influenced by friends
    def social_influence(self, hu, long_short_term):
        hu = torch.transpose(hu, 0, 1)
        outputs = []
        support_sizes = [1, self.samples_2, self.samples_1 * self.samples_2]
        num_samples = [self.samples_1, self.samples_2]
        for i in range(self.max_length):
            count = 0
            hu_ = hu[i]  # implement 1 of 20
            hidden = [hu_, long_short_term[0], long_short_term[1]]
            for layer in self.layers:
                next_hidden = []
                for hop in range(self.num_layers - count):
                    neigh_dims = [self.batch_size * support_sizes[hop],
                                  num_samples[self.num_layers - hop - 1],
                                  self.embedding_size]
                    h = layer([hidden[hop],
                               torch.reshape(hidden[hop + 1], neigh_dims)])
                    next_hidden.append(h)
                hidden = next_hidden
                count += 1
            outputs.append(hidden[0])
        feat = torch.stack(outputs, axis=0)
        # hu.shape, feat.shape : [max_length, batch, embedding_size]

        sr = self.W2(torch.cat((hu, feat), dim=2))  # final representation

        # return : [batch, max_length, item_embedding]
        return sr

    # get item score
    def score(self, sr, mask_y):
        logits = sr @ self.item_embedding(self.item_indices).t()  # similarity
        # logit shape : [max_length, batch, item_embedding]

        mask = mask_y.long()
        logits = torch.transpose(logits, 0, 1)
        logits *= torch.unsqueeze(mask, 2)

        return logits

    def forward(self, feed_dict):
        '''
        * Individual interest
            - Input_x: Itemid that user consumed in Timeid(session) - input data
                [batch_size, max_length]
            - Input_y: Itemid that user consumed in Timeid(session) - label
                [batch_size, max_length]
            - mask_y: mask of input_y
                [batch_size, max_length]
        * Friends' interest (long-term)
            - support_nodes_layer1: Userid of friends' friends
                [batch_size * samples_1 * samples_2]
            - support_nodes_layer2: Userid of friends
                [batch_size * samples_2]
        * Friends' interest (short-term)
            - support_sessions_layer1: Itemid that friends' friends spent most recently on Timeid.
                [batch_size * samples_1 * samples_2]
            - support_sessions_layer2: Itemid that friends spent most recently on Timeid.
                [batch_size * samples_2]
            - support_lengths_layer1: Number of items consumed by support_sessions_layer1
                [batch_size * samples_1 * samples_2]
            - support_lengths_layer2: Number of items consumed by support_sessions_layer2
                [batch_size * samples_2]
        '''
        labels = feed_dict['output_session']

        # interest,feed_dict['input_session']为个人的会话,output_session含义未知
        hu = self.individual_interest(feed_dict['input_session'],feed_dict['input_pop'])
        long_short_term = self.friends_interest(feed_dict['support_nodes_layer1'],
                                                feed_dict['support_nodes_layer2'],
                                                feed_dict['support_sessions_layer1'],
                                                feed_dict['support_sessions_layer2'],
                                                feed_dict['layer1_followee'],
                                                feed_dict['layer2_followee'],
                                                feed_dict['session1_pop'],
                                                feed_dict['session2_pop'],
                                                feed_dict['mask_y'])

        # social influence
        sr = self.social_influence(hu, long_short_term)

        # score
        logits = self.score(sr, feed_dict['mask_y'])

        # metric
        recall = self._recall(logits, labels)
        ndcg = self._ndcg(logits, labels, feed_dict['mask_y'])

        # loss
        logits = (torch.transpose(logits, 1, 2)).to(dtype=torch.float)  # logits : [batch, item_embedding, max_length]
        labels = labels.long()  # labels : [batch, max_length]

        loss = F.cross_entropy(logits, labels)

        return loss, recall.item(), ndcg.item()  # loss, recall_k, ndcg

    def predict(self, feed_dict):
        labels = feed_dict['output_session']

        hu = self.individual_interest(feed_dict['input_session'],feed_dict['input_pop'])

        long_short_term = self.friends_interest(feed_dict['support_nodes_layer1'],
                                                feed_dict['support_nodes_layer2'],
                                                feed_dict['support_sessions_layer1'],
                                                feed_dict['support_sessions_layer2'],
                                                feed_dict['layer1_followee'],
                                                feed_dict['layer2_followee'],
                                                feed_dict['session1_pop'],
                                                feed_dict['session2_pop'],
                                                feed_dict['mask_y'])


        sr = self.social_influence(hu, long_short_term)

        logits = self.score(sr, feed_dict['mask_y'])

        # metric
        recall = self._recall(logits, labels)
        ndcg = self._ndcg(logits, labels, feed_dict['mask_y'])

        # loss
        logits = (torch.transpose(logits, 1, 2)).to(dtype=torch.float)  # logits : [batch, item_embedding, max_length]
        labels = labels.long()  # labels : [batch, max_length]

        loss = F.cross_entropy(logits, labels)

        return loss, recall.item(), ndcg.item()

    def _recall(self, predictions, labels):
        batch_size = predictions.shape[0]
        _, top_k_index = torch.topk(predictions, k=20, dim=2)  # top_k_index : [batch, max_length, k]

        labels = labels.long()
        labels = torch.unsqueeze(labels, dim=2)  # labels : [batch, max_length, 1]
        corrects = (top_k_index == labels) * (labels != 0)  # corrects : [batch, max_length, k]
        recall_corrects = torch.sum(corrects, dim=2).to(dtype=torch.float)  # corrects : [batch, max_length]

        mask_sum = (labels != 0).sum(dim=1)  # mask_sum : [batch, 1]
        mask_sum = torch.squeeze(mask_sum, dim=1)  # mask_sum : [batch]

        recall_k = (recall_corrects.sum(dim=1) / mask_sum).sum()

        return recall_k / batch_size

    def _ndcg(self, logits, labels, mask):
        num_items = logits.shape[2]
        logits = torch.reshape(logits, (logits.shape[0] * logits.shape[1], logits.shape[2]))
        predictions = torch.transpose(logits, 0, 1)

        labels = labels.long()
        targets = torch.reshape(labels, [-1])
        pred_values = torch.unsqueeze(torch.diagonal(predictions[targets]), -1)
        # tile_pred_values = torch.tile(pred_values, [1, num_items])
        tile_pred_values = pred_values.repeat(1, num_items)
        ranks = torch.sum((logits > tile_pred_values).type(torch.float), -1) + 1
        ndcg = 1. / (torch.log2(1.0 + ranks))

        mask_sum = torch.sum(mask)
        mask = torch.reshape(mask, [-1])
        ndcg *= mask

        return torch.sum(ndcg) / mask_sum