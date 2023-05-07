#coding:utf-8
import pandas as pd
import numpy as np
import os
import datetime

# 用于生成datatype_dataset.csv文件
# 即sessions
def get_sessions(start, end):
    d1 = datetime.datetime.strptime(start, "%Y%m%d")
    d2 = datetime.datetime.strptime(end, "%Y%m%d")
    d = d2 - d1
    return d.days

class generate(object):

    def __init__(self, dataPath, sessPath):
        self._data = pd.read_csv(dataPath,dtype={'time':int})
        self.sessPath = sessPath
        self.sess_count = 0


    def stati_data(self):
        print('总数据量:', len(self._data))
        print('总session数:', len(self._data.drop_duplicates(['use_ID', 'time'])))
        print('平均session长度:', len(self._data) / len(self._data.drop_duplicates(['use_ID', 'time'])))
        print('总user数:', len(self._data.drop_duplicates('use_ID')))
        print('平均每个用户拥有的session个数:',
              len(self._data.drop_duplicates(['use_ID', 'time'])) / len(self._data.drop_duplicates('use_ID')))
        print('总item数:', len(self._data.drop_duplicates('ite_ID')))

        print('数据集时间跨度：', min(self._data.time), '~', max(self._data.time))

    def reform_u_i_id(self):
        # 将数据中的item和user重新编号，然后再生成session
        user_to_id = {}
        item_to_id = {}
        # 对user进行重新编号
        user_count = 0
        item_count = 0
        session_count = 0
        # for i in range(len(self._data)):
        #     self._data.at[i,'time'] = self._data.at[]


        time_id = self._data.drop_duplicates('time')['time']
        time_ranks = sorted(time_id)


        for i in range(len(self._data)):
            # 对user 和 item同时进行重新编号
            u_id = self._data.at[i, 'use_ID']
            i_id = self._data.at[i, 'ite_ID']
            s_id = self._data.at[i, 'time']
            if u_id in user_to_id.keys():
                self._data.at[i, 'use_ID'] = user_to_id[u_id]
            else:
                user_to_id[u_id] = user_count
                self._data.at[i, 'use_ID'] = user_count
                user_count += 1
            if i_id in item_to_id.keys():
                self._data.at[i, 'ite_ID'] = item_to_id[i_id]
            else:
                item_to_id[i_id] = item_count
                self._data.at[i, 'ite_ID'] = item_count
                item_count += 1
            self._data.at[i, 'time'] = time_ranks.index(s_id)


        # self._data.to_csv('/home/yunzhe/sq/data/middle_data.csv', index=False)
        self._data.to_csv('data/middle_data.csv', index=False)

        self.sess_count = len(time_ranks)
        print('user_count', user_count)
        print('item_count', item_count)
        print('session_count', self.sess_count + 1)
    # 按照实验设计，test的session是从数据集的最后一个月随机抽取百分之二十的session得到的
    # TallM中使用的test集合是每个用户最后一个session
    def generate_train_test_session(self):
        self.stati_data()  # 统计数据集
        self.reform_u_i_id()  # 重新编码user和item
        # self._data = pd.read_csv('/home/yunzhe/sq/data/middle_data.csv')
        self._data = pd.read_csv('data/middle_data.csv')

        session_path = self.sessPath
        if os.path.exists(session_path):
            os.remove(session_path)
        session_file = open(session_path, 'a')
        maxL=0
        # 这里最好使用numpy的格式，最后也按照这样的格式进行保存
        user_num = len(self._data['use_ID'].drop_duplicates())
        item_num = len(self._data['ite_ID'].drop_duplicates())
        session_file.write(str(user_num) + ',' + str(item_num) + ',' + str(self.sess_count) + '\n')
        last_userid = self._data.at[0, 'use_ID']
        last_time = self._data.at[0, 'time']
        session = str(last_userid) + ',' + str(self._data.at[0, 'ite_ID'])
        temp = 1
        for i in range(1, len(self._data)):
            # 文件使用降序打开
            # 最终session的格式为user_id,item_id:item_id...@item_id:item_id...@...
            userid = self._data.at[i, 'use_ID']
            itemid = self._data.at[i, 'ite_ID']
            time = self._data.at[i, 'time']
            if userid == last_userid and time == last_time:
                # 需要将session写入到文件中，然后开始
                session += ":" + str(itemid)
                temp+=1
            elif userid != last_userid:
                session_file.write(session + ';' + str(last_time) + '\n')
                last_userid = userid
                last_time = time
                session = str(userid) + ',' + str(itemid)
                temp += 1
            else:
                session += ';' + str(last_time) + '@' + str(itemid)
                last_time = time
                maxL = max(maxL, temp)
                temp=1
        print(maxL)


if __name__ == '__main__':
    # datatype = ['tallM', 'gowalla','amazon','ml-1m','foursquare']
    # D_id = 1
    # dataname = datatype[D_id]
    # dataPath = '/home/yunzhe/sq/data/' + dataname + '_data.csv'
    # sessPath = '/home/yunzhe/sq/data/' + dataname + '_dataset.csv'
    # # pd.read_csv(dataPath)
    # object = generate(dataPath, sessPath)
    # object.generate_train_test_session()

    datatype = ['tallM', 'gowalla','amazon','ml-1m','foursquare','Amazon_Instant_Video']
    D_id = 5
    dataname = datatype[D_id]
    dataPath = 'data/' + dataname + '_data.csv'
    sessPath = 'data/' + dataname + '_dataset.csv'
    # pd.read_csv(dataPath)
    object = generate(dataPath, sessPath)
    object.generate_train_test_session()