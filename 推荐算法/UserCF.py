import math
import pandas as pd
import numpy as np
import random

from collections import defaultdict
from operator import itemgetter


def LoadMovieLensData(filepath, train_rate):
    """
    加载数据，制作训练集样本和测试集样本，返回训练集样本的user-item表和测试集样本的user-item表
    :param filepath: 读取的文件路径
    :param train_rate:  控制训练集样本占的比重
    :return: 返回训练集样本的user-item和测试集样本的user-item表
    """
    data = pd.read_table(filepath, sep='::', header=None,
                         names=['UserID', 'MovieID', 'Rating', 'TimeStamp'], engine='python')
    data = data[['UserID', 'MovieID']]
    train = []
    test = []
    for idx, row in data.iterrows():
        user = int(row['UserID'])
        item = int(row['MovieID'])
        if random.random() < train_rate:  # random.random返回的是一个在0-1之间的随机小数，所以设置这个值可以大概控制训练集样本和测试集样本的分布
            train.append([user, item])
        else:
            test.append([user, item])
    return PreProcessData(train), PreProcessData(test)


def PreProcessData(originData):
    """
    originData输入的表是[[user,movie]]有两层列表
    建立user-item表{
        {'user1':{movie1,movie2}},
        {'user2':{movie1,movie3}}
    }
    :param originData:
    :return:
    """
    trainData = dict()
    for user, item in originData:
        trainData.setdefault(user, set())  # 字典为没有这个user的时候默认加入空集合
        trainData[user].add(item)
    return trainData


class UserCF(object):
    def __init__(self, trainData, similarity='cosine'):
        self._trainData = trainData  # user-item
        self._similarity = similarity
        self._userSimMatrix = dict()

    def similarity(self):
        # 建立item-user表
        item_user = dict()
        for user, items in self._trainData.items():
            # 注意这里的items是一个集合，没有重复物品，需要遍历一遍这里的item，找到每个item有哪些人
            for item in items:
                item_user.setdefault(item, set())
                item_user[item].add(user)  # 有了物品当key，我就需要把人作为value加入到集合当中
        # 根据item-user表计算用户物品交叉表
        for item, users in item_user.items():
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    self._userSimMatrix.setdefault(u, defaultdict(int))  # 初始化当字典中的键不存在的时候，值是0
                    if self._similarity == 'cosine':  # 余弦相似度
                        self._userSimMatrix[u][v] += 1  # 将u和v喜欢的物品 + 1
                    elif self._similarity == 'iif':  # 加了惩罚项的余弦相似度
                        self._userSimMatrix[u][v] += 1. / math.log(1 + len(users))
        # 计算用户相似度表_userSixMatrix
        for u, related_users in self._userSimMatrix.items():
            for v, cuv in related_users.items():
                cu = len(self._trainData[u])  # 用户u有多少喜欢的物品数
                cv = len(self._trainData[v])  # 用户v有多少喜欢的物品数
                self._userSimMatrix[u][v] = cuv * np.sqrt(cu * cv)

    def recommend(self, user, N, K):
        """
        用户u对物品的感兴趣程度排序，获取前N个物品进行推荐
            p(u,i) = ∑WuvRvi
            其中Wuv是用户u和用户v的相似度，Rvi是用户v对物品i的感兴趣程度，因为采取的都是用户喜欢的反馈数据，所以为1
            要计算u对物品i的相似度，首先需要找到前K个相似的用户，之后找K个相似用户喜爱的物品并且用户u还没有反馈过，
            根据K个用户的相似程度来计算对物品i的感兴趣程度
        :param self:
        :param user: 被推荐的用户数
        :param N: 推荐的商品数
        :param K: 前K个相似的用户
        :return:
        """
        recommends = dict()  # 推荐物品的字典
        related_items = self._trainData[user]  # 用户user反馈的物品数
        for v, sim in sorted(self._userSimMatrix[user].items(), key=itemgetter(1), reverse=True)[:K]:
            for item in self._trainData[v]:
                if item in related_items:
                    continue
                recommends.setdefault(item, 0.)
                recommends[item] += sim  # 用户v对物品喜欢（1） * 用户u与用户v之间的相似度
        # 根据用户u对被推荐物品的感兴趣程度，获得前N个
        return dict(sorted(recommends.items(), key=itemgetter(1), reverse=True)[:N])

    def train(self):
        self.similarity()


if __name__ == '__main__':
    train, test = LoadMovieLensData(filepath='./code/ml-1m/ratings.dat', train_rate=0.8)
    print('train size %d, test size %d' % (len(train), len(test)))
    UserCF = UserCF(trainData=train)
    UserCF.train()
    print(UserCF.recommend(list(test.keys())[0], 5, 20))
