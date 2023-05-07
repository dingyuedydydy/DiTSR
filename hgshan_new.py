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
from data_loader import data_generation
import scipy.sparse as sp
import argparse

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class HGSHAN():
    # data_type :  TallM / GWL
    def __init__(self, args, logger):
        print('init ... ')
        self.input_data_type = args.dataset

        self.logger = logger

        self.dg = data_generation(self.input_data_type)
        # 数据格式化
        self.dg.gen_train_test_data()
        self.norm_adj_s, self.mean_adj_s = self.dg.get_adj_mat('s')
        # self.norm_adj_i, self.mean_adj_i = self.dg.get_adj_mat('i')
        self.train_user_purchased_item_dict = self.dg.user_purchased_item

        self.user_number = self.dg.user_number
        self.item_number = self.dg.item_number
        self.neg_number = self.dg.neg_number
        self.session_number = self.dg.session_number
        self.ml = self.dg.max_len
        self.sample_number = 50
        self.test_users = self.dg.test_users
        self.test_candidate_items = self.dg.test_candidate_items
        self.test_sessions = self.dg.test_sessions
        self.test_pre_sessions = self.dg.test_pre_sessions
        self.test_real_items = self.dg.test_real_items

        self.global_dimension = args.dim
        self.batch_size = args.batch_size
        self.K = 20
        self.results = []  # 可用来保存test每个用户的预测结果，最终计算precision
        self.weight_size = [args.dim]
        self.n_layers = len(self.weight_size)
        self.step = 0
        self.iteration = args.n_epochs
        self.lamada_u_v = args.lamada_u_v
        self.lr = args.lr
        self.lamada_a = args.lamada_a
        self.weights = self._init_weights()
        self.mess_dropout = [0.7]
        self.initializer = tf.random_normal_initializer(mean=0, stddev=0.01)
        self.initializer_param = tf.random_uniform_initializer(minval=-np.sqrt(3 / self.global_dimension),
                                                               maxval=np.sqrt(3 / self.global_dimension))

        self.user_id = tf.placeholder(tf.int32, shape=[None], name='user_id')
        self.item_id = tf.placeholder(tf.int32, shape=[None], name='item_id')

        # 不管是当前的session，还是之前的session集合，在数据处理阶段都是一个数组，数组内容为item的编号
        self.current_session = tf.placeholder(tf.int32, shape=[None], name='current_session')
        self.current_time = tf.placeholder(tf.int32, shape=[None], name='current_times')
        self.pre_time = tf.placeholder(tf.int32, shape=[None], name='pre_times')
        self.pre_sessions = tf.placeholder(tf.int32, shape=[None, None, self.ml], name='pre_sessions')
        self.neg_item_id = tf.placeholder(tf.int32, shape=[None], name='neg_item_id')

        # graph parameter.
        self.user_W = tf.get_variable('user_w', initializer=self.initializer_param,
                                               shape=[1,self.sample_number])
        self.item_W = tf.get_variable('item_w', initializer=self.initializer_param,
                                               shape=[1,self.sample_number])
        self.user_bias = tf.get_variable('user_bias', initializer=self.initializer_param,
                                               shape=[self.global_dimension])
        self.item_bias = tf.get_variable('item_bias', initializer=self.initializer_param,
                                                       shape=[self.global_dimension])

        # When aggregates user,item, and session, set sess_c shape=[3,self.global_dimension], and set sess_w shape=[1,3]
        # self.sess_c = tf.get_variable('sess_c', initializer=self.initializer_param,
        #                                                shape=[2,self.global_dimension])
        self.sess_c = tf.get_variable('sess_c', initializer=self.initializer_param,
                                                       shape=[3,self.global_dimension])
        # self.sess_W = tf.get_variable('sess_w', initializer=self.initializer_param,
        #                                                shape=[1,2])
        self.sess_W = tf.get_variable('sess_w', initializer=self.initializer_param,
                                                       shape=[1,3])
        ##############

        self.sess_bias =  tf.get_variable('sess_bias', initializer=self.initializer_param,
                                                       shape=[self.global_dimension])

        self.user_embedding_matrix = tf.get_variable('user_embedding_matrix', initializer=self.initializer,
                                                     shape=[self.user_number, self.global_dimension])
        self.item_embedding_matrix = tf.get_variable('item_embedding_matrix', initializer=self.initializer,
                                                     shape=[self.item_number, self.global_dimension])
        self.session_embedding_matrix = tf.get_variable('session_embedding_matrix', initializer=self.initializer,
                                                        shape=[self.session_number, self.global_dimension])
        self.the_second_w = tf.get_variable('the_second_w', initializer=self.initializer_param,
                                            shape=[self.global_dimension, self.global_dimension])  # 20 * 20
        self.the_second_bias = tf.get_variable('the_second_bias', initializer=self.initializer_param,
                                               shape=[self.global_dimension])

    def attention_level_zero_self(self,pre_sessions_embedding):
        att = tf.layers.dense(pre_sessions_embedding, self.global_dimension, activation=tf.nn.sigmoid,
                              kernel_initializer=self.initializer_param)
        att = tf.reshape(att, [self.batch_size, -1, self.global_dimension])
        zero_a = tf.matmul(att, tf.expand_dims(pre_sessions_embedding, -1))
        zero_a = tf.reshape(zero_a, [self.batch_size, -1, self.ml])
        paddings = -9e15 * tf.ones_like(zero_a)
        zero_a = tf.where(self.pre_sessions >= 0, zero_a, paddings)
        self.weight_zero = tf.nn.softmax(zero_a, axis=2)

        out = tf.reduce_sum(tf.multiply(pre_sessions_embedding, tf.expand_dims(self.weight_zero, axis=-1)),
                            axis=2)  # 1 * ? *100
        return out

    def attention_level_zero(self, user_embedding, pre_sessions_embedding):

        # first level attention
        att = tf.layers.dense(pre_sessions_embedding, self.global_dimension, activation=tf.nn.sigmoid,
                              kernel_initializer=self.initializer_param)
        att = tf.reshape(att, [self.batch_size, -1, self.global_dimension])
        zero_a = tf.matmul(att, tf.expand_dims(user_embedding, -1))
        zero_a = tf.reshape(zero_a, [self.batch_size, -1, self.ml])
        paddings = -9e15 * tf.ones_like(zero_a)
        zero_a = tf.where(self.pre_sessions >= 0, zero_a, paddings)
        self.weight_zero = tf.nn.softmax(zero_a, axis=2)
        out = tf.reduce_sum(tf.multiply(pre_sessions_embedding, tf.expand_dims(self.weight_zero, axis=-1)),
                            axis=2)
        return out

    def attention_level_one(self, user_embedding, att_sessions_embedding):
        # second level attention
        att = tf.layers.dense(att_sessions_embedding, self.global_dimension, activation=tf.nn.sigmoid,
                              kernel_initializer=self.initializer_param)
        self.weight_one = tf.nn.softmax(tf.matmul(att, tf.expand_dims(user_embedding, -1)), axis=1)
        out = tf.reduce_sum(tf.multiply(att_sessions_embedding, self.weight_one), axis=1)
        return out

    def attention_level_two(self, user_embedding, long_user_embedding, current_session_embedding, the_second_w,
                            the_second_bias):
        # 需要将long_user_embedding加入到current_session_embedding中来进行attention，
        # 论文中规定，long_user_embedding的表示也不会根据softmax计算得到的参数而变化。
        # third level attention
        self.weight = tf.nn.softmax(tf.transpose(tf.matmul(
            tf.nn.sigmoid(tf.add(
                tf.matmul(tf.concat([current_session_embedding, long_user_embedding], 0),
                          the_second_w),
                the_second_bias)), tf.transpose(user_embedding))))
        out = tf.reduce_sum(
            tf.multiply(tf.concat([current_session_embedding, long_user_embedding], 0),
                        tf.transpose(self.weight)), axis=0)
        return out

    def _create_gcn_embed_t(self):
        # 这里是采用GCN的方式进行聚合,包含session信息
        A_fold_hat = self._split_A_hat(self.norm_adj_s, self.session_number)
        embeddings = tf.concat([self.session_embedding_matrix, self.user_embedding_matrix, self.item_embedding_matrix],
                               axis=0)
        all_embeddings = []
        for k in range(0, self.n_layers):
            embeddings = tf.sparse_tensor_dense_matmul(A_fold_hat, embeddings)
            embeddings = tf.nn.leaky_relu(
                tf.matmul(embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout[k])
            all_embeddings += [embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)
        return all_embeddings

    def sample_neighbor(self,nodes, type):
        # 节点采样 type: u, i 区分user, item
        if len(nodes) == 0:
            _sample = np.zeros(self.sample_number,dtype=np.int)
        elif len(nodes) < self.sample_number:
            _sample = list(np.zeros((self.sample_number-len(nodes)),dtype=np.int))
            _sample = np.concatenate([_sample, nodes])
        else:
            _sample = np.random.choice(nodes,self.sample_number,replace=False)
        if type == 'i':
            sample = tf.nn.embedding_lookup(self.item_embedding_matrix, _sample)
        elif type == 'u':
            sample = tf.nn.embedding_lookup(self.user_embedding_matrix, _sample)

        return sample

    def _create_sage_embed_t(self):
        # 这里是采用Graphsage 采样的方式进行聚合
        # 注释部分为更改对item采样的内容，默认是进行user聚合
        all_embedings = self.session_embedding_matrix
        for k in range(0, self.n_layers):
            all_emb = []
            # session_sample = np.random.choice(np.arange(self.session_number),50,replace=False)
            # sample...
            neighbors_i = []
            neighbors_u = []
            sess_embs = []
            for gcn_layer in range(1):
                for i in range(self.session_number):

                    # 信息传递
                    ## 采样user
                    neighbor_u = self.sample_neighbor(self.dg.R[i].nonzero()[1],'u')
                    # 采样item
                    neighbor_i = self.sample_neighbor(self.dg.R_i[i].nonzero()[1],'i')

                    sess_emb = tf.nn.embedding_lookup(self.session_embedding_matrix,i)
                    sess_emb = tf.reshape(sess_emb, [1, -1])

                    neighbors_u += [neighbor_u]
                    sess_embs += [sess_emb]

                    user_agg = tf.nn.leaky_relu(
                        tf.matmul(self.user_W,neighbor_u)+self.user_bias
                    )

                    # 采样item
                    neighbors_i += [neighbor_i]

                    ### remove
                    # neighbors_i = tf.concat(neighbors_u,0)
                    ##

                    item_agg = tf.nn.leaky_relu(
                        tf.matmul (self.item_W,neighbor_i) + self.item_bias
                    )

                    con_input = tf.multiply(self.sess_c, tf.concat([sess_emb, user_agg, item_agg], 0))
                    # 聚合激活

                    # con_input = tf.multiply(self.sess_c, tf.concat([sess_emb, user_agg ], 0))
                    sess_sage_emb = tf.nn.leaky_relu(
                        tf.matmul(self.sess_W,con_input)+self.sess_bias)
                    sess_sage_emb = tf.nn.dropout(sess_sage_emb, 1 - self.mess_dropout[k])
                    all_embedings[i].assign(sess_sage_emb)
                # all_embedings[session_sample] = all_emb
        return all_embedings

    def _create_gcn_embed_i(self):

        A_fold_hat = self._split_A_hat(self.norm_adj_i, self.item_number)
        embeddings = tf.concat([self.item_embedding_matrix, self.se_embedding_matrix], axis=0)
        all_embeddings = []
        for k in range(0, self.n_layers):
            embeddings = tf.sparse_tensor_dense_matmul(A_fold_hat, embeddings)
            embeddings = tf.nn.leaky_relu(
                tf.matmul(embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout[k])
            all_embeddings += [embeddings]
        all_embeddings = tf.concat(all_embeddings, 1)

        return all_embeddings

    def _init_weights(self):
        all_weights = dict()

        initializer = tf.contrib.layers.xavier_initializer()

        self.weight_size_list = [self.global_dimension] + self.weight_size
        for k in range(self.n_layers):
            all_weights['W_gc_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_gc_%d' % k)

            all_weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

            all_weights['W_mlp_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_mlp_%d' % k)

        return all_weights

    def _split_A_hat(self, X, end):
        A_fold_hat = self._convert_sp_mat_to_sp_tensor(X[:end])
        return A_fold_hat

    # sparse matrix to sparse tensor
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _contrast_loss(self):
        # Foursqure 249 days, Gowalla 205 days, Amazon 613 weeks
        n_session = 205
        n_ran_con_sample = 10
        sim_neg_acc = tf.constant(0,dtype=float)
        pos_id = random.randint(0, n_session-1)
        pos_pair_id = pos_id + 1
        pos_id = tf.convert_to_tensor(pos_id)
        pos_pair_id = tf.convert_to_tensor(pos_pair_id)
        pos_id_emb = tf.nn.embedding_lookup(self.se_embedding_matrix, pos_id)
        pos_pair_id_emb = tf.nn.embedding_lookup(self.se_embedding_matrix, pos_pair_id)
        sim_pos = tf.exp(tf.multiply(pos_id_emb,pos_pair_id_emb)/0.1)
        sim_pos = tf.reduce_sum(sim_pos)
        for i in range(n_ran_con_sample):
            neg_id = random.randint(0, n_session)
            neg_id = tf.convert_to_tensor(neg_id)
            neg_id_emb = tf.nn.embedding_lookup(self.se_embedding_matrix, neg_id)
            sim_neg = tf.multiply(pos_id_emb,neg_id_emb)
            sim_neg = tf.exp(tf.reduce_sum(sim_neg)/0.1)
            sim_neg_acc +=sim_neg
        cont_loss = -tf.log(sim_pos/(sim_pos+sim_neg_acc))

        return cont_loss

    def build_model(self):
        # 底层生成graph embedding的方式（GCN, SAGE)
        self.se_embedding_matrix = self._create_sage_embed_t()
        # self.se_embedding_matrix = self._create_gcn_embed_t()
        # self.se_embedding_matrix = self.session_embedding_matrix
        self.pre_time_embedding = tf.nn.embedding_lookup(self.se_embedding_matrix, self.pre_time)
        self.time_embedding = tf.nn.embedding_lookup(self.se_embedding_matrix, self.current_time)
        self.user_embedding = tf.nn.embedding_lookup(self.user_embedding_matrix, self.user_id)
        self.item_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.item_id)

        self.current_session_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.current_session)
        self.pre_sessions_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.pre_sessions)
        self.neg_item_embedding = tf.nn.embedding_lookup(self.item_embedding_matrix, self.neg_item_id)

        self.pre_sessions_embedding = tf.add(self.pre_sessions_embedding, tf.reshape(self.pre_time_embedding,
                                                                                     [self.batch_size, -1, 1,
                                                                                      self.global_dimension]))
        # 上层attention部分
        self.att_sessions_embedding = self.attention_level_zero(self.user_embedding, self.pre_sessions_embedding)
        self.long_user_embedding = self.attention_level_one(self.user_embedding, self.att_sessions_embedding)
        self.current_session_embedding = tf.add(self.current_session_embedding, self.time_embedding)
        self.hybrid_user_embedding = self.attention_level_two(self.user_embedding, self.long_user_embedding,
                                                              self.current_session_embedding,
                                                              self.the_second_w, self.the_second_bias)
        # 计算用户偏好
        self.positive_element_wise = tf.matmul(tf.expand_dims(self.hybrid_user_embedding, axis=0),
                                               tf.transpose(self.item_embedding + self.time_embedding))

        self.negative_element_wise = tf.matmul(tf.expand_dims(self.hybrid_user_embedding, axis=0),
                                               tf.transpose(self.neg_item_embedding + self.time_embedding))

        self.intention_loss = tf.reduce_mean(
            -tf.log(tf.nn.sigmoid(self.positive_element_wise - self.negative_element_wise)))
        self.regular_loss_u_v = self.lamada_u_v * tf.nn.l2_loss(self.user_embedding) + \
                                self.lamada_u_v * tf.nn.l2_loss(self.item_embedding) + \
                                self.lamada_u_v * tf.nn.l2_loss(self.time_embedding)
        self.regular_loss_a = self.lamada_a * tf.nn.l2_loss(self.the_second_w)
        self.regular_loss = tf.add(self.regular_loss_a, self.regular_loss_u_v)
        self.intention_loss = tf.add(self.intention_loss, self.regular_loss)

        ## Add contrastive loss
        self.cont_loss = 0.001*self._contrast_loss()
        self.intention_loss = tf.add(self.intention_loss, self.cont_loss)

        # 增加test操作，由于每个用户pre_sessions和current_session的长度不一样，
        # 所以无法使用同一个矩阵进行表示同时计算，因此每个user计算一次，将结果保留并进行统计
        # 注意，test集合的整个item_embeeding得到的是 [M*K]的矩阵，M为所有item的个数，K为维度
        self.top_value, self.top_index = tf.nn.top_k(self.positive_element_wise, k=self.K, sorted=True)

    def run(self):
        print('running ... ')
        gpu_config = tf.ConfigProto()
        gpu_config.gpu_options.allow_growth = True
        with tf.Session(config=gpu_config) as self.sess:
            self.intention_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(
                self.intention_loss)
            init = tf.global_variables_initializer()
            self.sess.run(init)
            for iter in range(self.iteration):
                print('new iteration begin ... ')
                print('iteration: ', str(iter))
                st = time()
                while self.step * self.batch_size < self.dg.records_number:
                    # 按批次读取数据
                    batch_user, batch_item, batch_session, batch_neg_item, batch_pre_sessions, batch_time, batch_pre_time = self.dg.gen_train_batch_data(
                        self.batch_size)

                    _, f_we, s_we, t_we = self.sess.run(
                        [self.intention_optimizer, self.weight_zero, self.weight_one, self.pre_sessions_embedding],
                        feed_dict={self.user_id: batch_user, self.item_id: batch_item,
                                   self.current_session: batch_session,
                                   self.neg_item_id: batch_neg_item,
                                   self.pre_sessions: batch_pre_sessions,
                                   self.current_time: batch_time,
                                   self.pre_time: batch_pre_time
                                   })

                    self.step += 1
                print('ite cost %s, eval ...' % (time() - st))
                self.evolution()
                print(self.step, '/', self.dg.train_batch_id, '/', self.dg.records_number)
                self.step = 0

    # 指标评测部分
    def recall_k(self, pre_top_k, true_items):
        # 同时返回Recall, MRR指标
        num = [5,10,20]
        right_pre = [0,0,0]
        mrr = [0,0,0]
        user_number = len(pre_top_k)
        for i in range(user_number):
            if true_items[i] in pre_top_k[i]:
                li = np.ravel(pre_top_k[i]).tolist()
                rank = li.index(true_items[i])+1
                for j in range(len(num)):
                    if rank<= num[j]:
                        right_pre[j] += 1
                        mrr[j] += 1.0 / (rank + 1.0)
        return np.array(right_pre) / len(true_items), np.array(mrr) / user_number

    def evolution(self):
        pre_top_k = []
        for _ in self.test_users:
            batch_user, batch_item, batch_session, batch_pre_session, batch_pre_times, batch_times = self.dg.gen_test_batch_data(
                self.batch_size)
            top_k_value, top_index = self.sess.run([self.top_value, self.top_index],
                                                   feed_dict={self.user_id: batch_user,
                                                              self.item_id: batch_item,
                                                              self.current_session: batch_session,
                                                              self.pre_sessions: batch_pre_session,
                                                              self.pre_time: batch_pre_times,
                                                              self.current_time: batch_times
                                                              })
            pre_top_k.append(top_index)
        recall, mrr = self.recall_k(pre_top_k, self.test_real_items)
        self.logger.info('recall@' + str(self.K) + ' = ' + str(recall))
        self.logger.info('mrr@' + str(self.K) + ' = ' + str(mrr))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='three', help='which model to evaluate')
    parser.add_argument('--dataset', type=str, default='gowalla', help='which dataset to use')
    # parser.add_argument('--dataset', type=str, default='Amazon_Instant_Video', help='which dataset to use')
    parser.add_argument('--dim', type=int, default=120, help='dimension of user and entity embeddings')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--n_epochs', type=int, default=10, help='the number of epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--lamada_u_v', type=float, default=0.00001, help='weight of luv regularization')
    parser.add_argument('--lamada_a', type=float, default=0.01, help='weight of La regularization')
    parser.add_argument('--ratio', type=float, default=1, help='size of training dataset')
    args = parser.parse_args()

    logging.config.fileConfig('logging.conf')
    logger = logging.getLogger()
    logger.info('model_type=%s, dataset=%s' % (args.model_type, args.dataset))
    logger.info(
        'dim=%d, lr=%.4f,  n_epochs=%d, batch_size=%d, lamada_u_v=%.4f, lamada_a=%.4f, ratio=%.2f'
        % (args.dim, args.lr, args.n_epochs, args.batch_size, args.lamada_u_v, args.lamada_a, args.ratio))
    model = HGSHAN(args, logger)
    model.build_model()
    model.run()

