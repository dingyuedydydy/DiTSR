import numpy as np
import pandas as pd
import tensorflow as tf
import random
import copy
import logging
import os
import sys
import logging.config
from time import time
import scipy.sparse as sp

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class data_generation():
    def __init__(self, type):
        print('init')
        self.data_type = type

        # self.dataset = '/home/yunzhe/sq/data/' + self.data_type + '_dataset.csv'
        # self.path = '/home/yunzhe/sq/data/' + self.data_type

        self.dataset = 'data/' + self.data_type + '_dataset.csv'
        self.path = 'data/' + self.data_type

        self.train_users = []
        self.train_sessions = []  # 当前的session
        self.traiitem_number = []  # 随机采样得到的positive
        self.train_neg_items = []  # 随机采样得到的negative
        self.train_pre_sessions = []  # 之前的session集合
        self.train_pre_time = []
        self.train_current_time = []

        self.test_users = []
        self.test_candidate_items = []
        self.test_sessions = []
        self.test_pre_sessions = []
        self.test_real_items = []
        self.test_pre_time = []
        self.test_current_time = []

        self.max_len = 5
        self.neg_number = 1
        self.user_number = 0
        self.item_number = 0
        self.session_number = 0
        self.train_batch_id = 0
        self.test_batch_id = 0
        self.records_number = 0
                                
    def gen_train_test_data(self):
        self.data = pd.read_csv(self.dataset, names=['user', 'sessions', 'sessionlens'], dtype='str')
        is_first_line = 1
        for line in self.data.values:
            if is_first_line:
                self.user_number = int(line[0])
                self.item_number = int(line[1])
                self.session_number = int(line[2])
                self.user_purchased_item = dict()  # 保存每个用户购买记录，可用于train时负采样和test时剔除已打分商品
                self.long_pad_item = dict()
                self.long_time = dict()
                self.R = sp.dok_matrix((self.session_number, self.user_number), dtype=np.float32)
                self.R_i = sp.dok_matrix((self.session_number, self.item_number), dtype=np.float32)
                is_first_line = 0
            else:
                temp_pre_session = []
                temp_pre_time = []
                first_temp = []
                user_id = int(line[0])
                sessions = [i for i in line[1].split('@')]
                size = len(sessions)
                f_session = sessions[0].split(';')
                the_first_session = [int(i) for i in f_session[0].split(':')]
                self.R[int(f_session[1]), user_id] = 1
                for k in range(len(the_first_session)):
                    self.R_i[int(f_session[1]), the_first_session[k]] = 1
                # 补齐，如果比max_len则从后往前截断，如果比max_len小则从前往后补齐
                if len(the_first_session) >= self.max_len:
                    first_temp = the_first_session[len(the_first_session) - self.max_len:]
                else:
                    first_temp = [-1] * self.max_len
                    for st in range(len(the_first_session)):
                        first_temp[self.max_len - len(the_first_session) + st] = the_first_session[st]
                temp_pre_session.extend(first_temp)
                temp_pre_time.append(int(f_session[1]))
                self.long_time[user_id] = [int(f_session[1])]
                self.long_pad_item[user_id] = first_temp
                self.train_pre_sessions.append(first_temp)
                self.train_pre_time.append([int(f_session[1])])
                tmp = copy.deepcopy(the_first_session)
                self.user_purchased_item[user_id] = tmp
                for j in range(1, size-1):
                    # 每个用户的每个session在train_users中都对应着其user_id，不一定是连续的
                    self.train_users.append(user_id)
                    # test = sessions[j].split(':')
                    n_session = sessions[j].split(';')
                    temp = []
                    current_session = [int(it) for it in n_session[0].split(':')]
                    self.R[int(n_session[1]), user_id] = 1
                    for k in range(len(current_session)):
                        self.R_i[int(n_session[1]), current_session[k]] = 1
                    # 补齐，如果比max_len则从后往前截断，如果比max_len小则从前往后补齐
                    if len(current_session)>=self.max_len:
                        temp = current_session[len(current_session) - self.max_len:]
                    else:
                        temp = [-1]*self.max_len
                        for st in range(len(current_session)):
                            temp[self.max_len -  len(current_session) + st] = current_session[st]
                    neg = self.gen_neg(user_id)
                    self.train_neg_items.append(neg)
                    # 将当前session加入到用户购买的记录当中
                    # 之所以放在这个位置，是因为在选择测试item时，需要将session中的一个item移除、
                    # 如果放在后面操作，当前session中其实是少了一个用来做当前session进行预测的item
                    temp_pre_session.extend(temp)
                    temp_pre_time.append(int(n_session[1]))
                    if j != size-2:
                        te = copy.deepcopy(temp_pre_session)
                        self.train_pre_sessions.append(te)
                        te_time = copy.deepcopy(temp_pre_time)
                        self.train_pre_time.append(te_time)
                    if j ==size-2:
                        self.long_pad_item[user_id] = temp_pre_session
                        self.long_time[user_id] = temp_pre_time
                    tmp = copy.deepcopy(current_session)
                    self.user_purchased_item[user_id].extend(tmp)
                    # 随机挑选一个作为prediction item
                    item = random.choice(current_session)
                    self.traiitem_number.append(item)
                    current_session.remove(item)
                    self.train_current_time.append([int(n_session[1])])
                    self.train_sessions.append(current_session)
                    self.records_number += 1
                self.test_users.append(user_id)
                test_se = sessions[size - 1].split(';')
                current_session = [int(it) for it in test_se[0].split(':')]
                item = random.choice(current_session)
                self.test_real_items.append(int(item))
                current_session.remove(item)
                self.test_sessions.append(current_session)
                self.test_pre_sessions.append(self.long_pad_item[user_id])
                self.test_pre_time.append(self.long_time[user_id])
                self.test_current_time.append([int(test_se[1])])
        # test集中每个用户的预测的候选集就是整个item集合
        self.test_candidate_items = list(range(self.item_number))

    def gen_test_data(self):
        self.data = pd.read_csv(self.test_dataset, names=['user', 'sessions'], dtype='str')
        self.test_candidate_items = list(range(self.item_number))

        # 对于ndarray进行sample得到test目标数据
        sub_index = self.shuffle(len(self.data.values))
        data = self.data.values[sub_index]

        for line in data:
            user_id = int(line[0])
            if user_id in self.user_purchased_item.keys():
                session_time = line[1].split(';')
                current_session = [int(i) for i in session_time[0].split(':')]
                self.test_users.append(user_id)
                item = random.choice(current_session)
                self.test_real_items.append(int(item))
                current_session.remove(item)
                self.test_sessions.append(current_session)
                self.test_pre_sessions.append(self.long_pad_item[user_id])
                self.test_pre_time.append(self.long_time[user_id])
                self.test_current_time.append([int(session_time[1])])

    def shuffle(self, test_length):
        index = np.array(range(test_length))
        np.random.shuffle(index)
        sub_index = np.random.choice(index, int(test_length * 0.2))
        return sub_index

    def gen_neg(self, user_id):
        neg_item = np.random.randint(self.item_number)
        while neg_item in self.user_purchased_item[user_id]:
            neg_item = np.random.randint(self.item_number)
        return neg_item

    def gen_train_batch_data(self, batch_size):
        # l = len(self.train_users)

        if self.train_batch_id == self.records_number:
            self.train_batch_id = 0

        batch_user = self.train_users[self.train_batch_id:self.train_batch_id + batch_size]
        batch_item = self.traiitem_number[self.train_batch_id:self.train_batch_id + batch_size]
        batch_session = self.train_sessions[self.train_batch_id]
        batch_time = self.train_current_time[self.train_batch_id]
        batch_pre_time = self.train_pre_time[self.train_batch_id]
        batch_neg_item = self.train_neg_items[self.train_batch_id:self.train_batch_id + batch_size]
        batch_pre_session = np.array(self.train_pre_sessions[self.train_batch_id]).reshape( batch_size, -1, self.max_len)

        self.train_batch_id = self.train_batch_id + batch_size

        return batch_user, batch_item, batch_session, batch_neg_item, batch_pre_session, batch_time, batch_pre_time

    def gen_test_batch_data(self, batch_size):
        l = len(self.test_users)

        if self.test_batch_id == l:
            self.test_batch_id = 0

        batch_user = self.test_users[self.test_batch_id:self.test_batch_id + batch_size]
        batch_item = self.test_candidate_items
        batch_session = self.test_sessions[self.test_batch_id]
        batch_pre_session = np.array(self.test_pre_sessions[self.test_batch_id]).reshape( batch_size, -1, self.max_len)
        batch_pre_time = self.test_pre_time[self.test_batch_id]
        batch_time = self.test_current_time[self.test_batch_id]

        self.test_batch_id = self.test_batch_id + batch_size

        return batch_user, batch_item, batch_session, batch_pre_session, batch_pre_time, batch_time




    def get_adj_mat(self, type):
        try:
            t1 = time()
            norm_adj_mat = sp.load_npz(self.path + '/'+ type+'_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/'+ type+'_mean_adj_mat.npz')
            print('already load adj matrix', norm_adj_mat.shape, time() - t1)

        except Exception:
            norm_adj_mat, mean_adj_mat = eval('self.create_adj_mat_' + type+'()')
            sp.save_npz(self.path + '/' + type + '_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/' + type + '_mean_adj_mat.npz', mean_adj_mat)
        return norm_adj_mat, mean_adj_mat

    def create_adj_mat_s(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.session_number + self.user_number+self.item_number, self.session_number + self.user_number+self.item_number), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R.tolil()
        R_i = self.R_i.tolil()

        adj_mat[:self.session_number, self.session_number:self.session_number+self.user_number] = R
        adj_mat[:self.session_number, self.session_number+self.user_number:self.session_number+self.user_number+self.item_number] = R_i
        adj_mat[self.session_number:self.session_number+self.user_number, :self.session_number] = R.T
        adj_mat[self.session_number + self.user_number:self.session_number + self.user_number+self.item_number, :self.session_number] = R_i.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        norm_adj_mat = self.normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = self.normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def create_adj_mat_i(self):
        t1 = time()
        adj_mat = sp.dok_matrix((self.session_number + self.item_number, self.session_number + self.item_number), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.R_i.tolil()

        adj_mat[:self.item_number, self.item_number:] = R
        adj_mat[self.item_number:, :self.item_number] = R.T
        adj_mat = adj_mat.todok()
        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        norm_adj_mat = self.normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = self.normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def normalized_adj_single(self, adj):
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        # norm_adj = adj.dot(d_mat_inv)
        print('generate single-normalized adjacency matrix.')
        return norm_adj.tocoo()


# class data_generation_PE(data_generation):
#     def __init__(self, type):
#         data_generation.__init__(self, type)
#         self.dataset = '/home/yunzhe/sq/data/' + self.data_type + '_dataset_PE.csv'
#
#     def gen_train_test_data(self):
#         self.data = pd.read_csv(self.dataset, names=['user', 'sessions', 'sessionlens'], dtype='str')
#         is_first_line = 1
#         for line in self.data.values:
#             if is_first_line:
#                 self.user_number = int(line[0])
#                 self.item_number = int(line[1])
#                 self.session_number = int(line[2])
#                 self.user_purchased_item = dict()  # 保存每个用户购买记录，可用于train时负采样和test时剔除已打分商品
#                 self.long_pad_item = dict()
#                 self.long_time = dict()
#                 self.R = sp.dok_matrix((self.session_number, self.user_number), dtype=np.float32)
#                 self.R_i = sp.dok_matrix((self.session_number, self.item_number), dtype=np.float32)
#                 is_first_line = 0
#             else:
#                 temp_pre_session = []
#                 temp_pre_time = []
#                 first_temp = []
#                 user_id = int(line[0])
#                 sessions = [i for i in line[1].split('@')]
#                 size = len(sessions)
#                 f_session = sessions[0].split(';')
#                 the_first_session = [i for i in f_session[0].split(':')]
#
#                 self.R[int(f_session[1]), user_id] = 1
#                 for k in range(len(the_first_session)):
#                     self.R_i[int(f_session[1]), the_first_session[k]] = 1
#                 # 补齐，如果比max_len大则从后往前截断，如果比max_len小则从前往后补齐
#                 if len(the_first_session) >= self.max_len:
#                     first_temp = the_first_session[len(the_first_session) - self.max_len:]
#                 else:
#                     first_temp = [-1] * self.max_len
#                     for st in range(len(the_first_session)):
#                         first_temp[self.max_len - len(the_first_session) + st] = the_first_session[st]
#
#                 temp_pre_session.extend(first_temp)
#                 temp_pre_time.append(int(f_session[1]))
#                 self.long_time[user_id] = [int(f_session[1])]
#                 self.long_pad_item[user_id] = first_temp
#                 self.train_pre_sessions.append(first_temp)
#                 self.train_pre_time.append([int(f_session[1])])
#                 tmp = copy.deepcopy(the_first_session)
#                 self.user_purchased_item[user_id] = tmp
#                 for j in range(1, size - 1):
#                     # 每个用户的每个session在train_users中都对应着其user_id，不一定是连续的
#                     self.train_users.append(user_id)
#                     # test = sessions[j].split(':')
#                     n_session = sessions[j].split(';')
#                     temp = []
#                     current_session = [int(it) for it in n_session[0].split(':')]
#                     self.R[int(n_session[1]), user_id] = 1
#                     for k in range(len(current_session)):
#                         self.R_i[int(n_session[1]), current_session[k]] = 1
#                     # 补齐，如果比max_len则从后往前截断，如果比max_len小则从前往后补齐
#                     if len(current_session) >= self.max_len:
#                         temp = current_session[len(current_session) - self.max_len:]
#                     else:
#                         temp = [-1] * self.max_len
#                         for st in range(len(current_session)):
#                             temp[self.max_len - len(current_session) + st] = current_session[st]
#                     neg = self.gen_neg(user_id)
#                     self.train_neg_items.append(neg)
#                     # 将当前session加入到用户购买的记录当中
#                     # 之所以放在这个位置，是因为在选择测试item时，需要将session中的一个item移除、
#                     # 如果放在后面操作，当前session中其实是少了一个用来做当前session进行预测的item
#                     temp_pre_session.extend(temp)
#                     temp_pre_time.append(int(n_session[1]))
#                     if j != size - 2:
#                         te = copy.deepcopy(temp_pre_session)
#                         self.train_pre_sessions.append(te)
#                         te_time = copy.deepcopy(temp_pre_time)
#                         self.train_pre_time.append(te_time)
#                     if j == size - 2:
#                         self.long_pad_item[user_id] = temp_pre_session
#                         self.long_time[user_id] = temp_pre_time
#                     tmp = copy.deepcopy(current_session)
#                     self.user_purchased_item[user_id].extend(tmp)
#                     # 随机挑选一个作为prediction item
#                     item = random.choice(current_session)
#                     self.traiitem_number.append(item)
#                     current_session.remove(item)
#                     self.train_current_time.append([int(n_session[1])])
#                     self.train_sessions.append(current_session)
#                     self.records_number += 1
#                 self.test_users.append(user_id)
#                 test_se = sessions[size - 1].split(';')
#                 current_session = [int(it) for it in test_se[0].split(':')]
#                 item = random.choice(current_session)
#                 self.test_real_items.append(int(item))
#                 current_session.remove(item)
#                 self.test_sessions.append(current_session)
#                 self.test_pre_sessions.append(self.long_pad_item[user_id])
#                 self.test_pre_time.append(self.long_time[user_id])
#                 self.test_current_time.append([int(test_se[1])])
#         # test集中每个用户的预测的候选集就是整个item集合
#         self.test_candidate_items = list(range(self.item_number))
#
#     def gen_test_data(self):
#         self.data = pd.read_csv(self.test_dataset, names=['user', 'sessions'], dtype='str')
#         self.test_candidate_items = list(range(self.item_number))
#
#         # 对于ndarray进行sample得到test目标数据
#         sub_index = self.shuffle(len(self.data.values))
#         data = self.data.values[sub_index]
#
#         for line in data:
#             user_id = int(line[0])
#             if user_id in self.user_purchased_item.keys():
#                 session_time = line[1].split(';')
#                 current_session = [int(i) for i in session_time[0].split(':')]
#                 self.test_users.append(user_id)
#                 item = random.choice(current_session)
#                 self.test_real_items.append(int(item))
#                 current_session.remove(item)
#                 self.test_sessions.append(current_session)
#                 self.test_pre_sessions.append(self.long_pad_item[user_id])
#                 self.test_pre_time.append(self.long_time[user_id])
#                 self.test_current_time.append([int(session_time[1])])

