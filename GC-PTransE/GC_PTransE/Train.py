import io
import os
import random
from typing import List, Tuple
from GC_PTransE.GlobalValue import GlobalValue
from GC_PTransE.Gradient import Gradient
from GC_PTransE.Utils import Utils


class Train:
    def __init__(self):
        self.fb_h = []  # 存储头实体编号的列表
        self.fb_l = []  # 存储尾实体编号的列表
        self.fb_r = []  # 存储关系编号的列表
        self.head_relation2tail = {}  # 存储头实体和关系对应的尾实体集合的字典
        self.fb_path2prob = []  # 存储路径和概率的列表

    def Write_Vec2File(self, file_name: str, vec: List[List[float]], number: int) -> None:
        '''
        将向量写入文件
        '''
        dir = os.path.dirname(file_name)
        if not os.path.exists(dir):
            os.makedirs(dir)

        with io.open(file_name, mode="w", encoding="UTF-8") as f:
            for i in range(number):
                for j in range(GlobalValue.vector_len):
                    str = "%.6f\t" % vec[i][j]
                    f.write(str)
                f.write("\n")
                f.flush()

    def random_tail(self, pos: int, neg_pos: int) -> int:
        '''
        随机替换尾实体
        '''
        key = (self.fb_h[pos], self.fb_r[pos])
        values = self.head_relation2tail.get(key, set())
        while neg_pos in values:
            neg_pos = random.randint(0, GlobalValue.entity_num - 1)
        return neg_pos

    def random_head(self, pos: int, neg_pos: int) -> int:
        '''
        随机替换头实体
        '''
        key = (neg_pos, self.fb_r[pos])
        values = self.head_relation2tail.get(key, set())
        while self.fb_l[pos] in values:
            neg_pos = random.randint(0, GlobalValue.entity_num - 1)
            key = (neg_pos, self.fb_r[pos])
            values = self.head_relation2tail.get(key, set())
        return neg_pos

    def random_relation(self, pos: int, neg_pos: int) -> int:
        '''
        随机替换关系
        '''
        key = (self.fb_h[pos], neg_pos)
        values = self.head_relation2tail.get(key, set())
        while self.fb_l[pos] in values:
            neg_pos = random.randint(0, GlobalValue.relation_num - 1)
            key = (self.fb_h[pos], neg_pos)
            values = self.head_relation2tail.get(key, set())
        return neg_pos

    def update_relation(self, pos) -> None:
        '''
        更新关系向量
        '''
        relation_neg = random.randint(0, GlobalValue.relation_num - 1)
        relation_neg = self.random_relation(pos, relation_neg)

        path2prob_list = self.fb_path2prob[pos]
        for path2prob in path2prob_list:
            path, prob = path2prob
            path_str = " ".join(str(pid) for pid in path)

            tmp_path2rel = (path_str, self.fb_r[pos])
            tmp_confidence = GlobalValue.path_confidence.get(tmp_path2rel, 0)
            tmp_confidence = (0.99 * tmp_confidence + 0.01) * prob  # 衰减和加权？？？？
            Gradient.train_path(self.fb_r[pos], relation_neg, path, tmp_confidence, self.loss)

    def bfgs(self, nepoch: int, nbatches: int) -> None:
        '''
        BFGS优化算法
        '''
        batchsize = len(self.fb_h) // nbatches  # 计算每次迭代处理的三元组数
        print("Batch size = %s" % batchsize)
        for epoch in range(nepoch):
            # loss function value
            self.loss = 0
            for batch in range(nbatches):
                for k in range(batchsize):
                    # random.randint方法返回指定范围内的整数  0-171
                    pos = random.randint(0, len(self.fb_h) - 1)  # 随机选取一行三元组, 行号
                    tmp_rand = random.randint(0, 99)

                    # 根据随机数决定采样替换类
                    if tmp_rand < 25:
                        # 负采样：尾实体替换
                        tail_neg = random.randint(0, GlobalValue.entity_num - 1)
                        tail_neg = self.random_tail(pos, tail_neg)
                        self.loss = Gradient.train_kb(self.fb_h[pos], self.fb_l[pos], self.fb_r[pos], self.fb_h[pos],
                                                      tail_neg,
                                                      self.fb_r[pos], self.loss)
                        # 归一化更新的负尾实体嵌入，以确保它们保持在合理的范围内
                        Utils.norm(GlobalValue.entity_vec[tail_neg], GlobalValue.vector_len)

                    elif tmp_rand < 50:
                        # 负采样：头实体替换
                        head_neg = random.randint(0, GlobalValue.entity_num - 1)
                        head_neg = self.random_head(pos, head_neg)
                        self.loss = Gradient.train_kb(self.fb_h[pos], self.fb_l[pos], self.fb_r[pos], head_neg,
                                                      self.fb_l[pos],
                                                      self.fb_r[pos], self.loss)
                        # 归一化更新的负头实体嵌入
                        Utils.norm(GlobalValue.entity_vec[head_neg], GlobalValue.vector_len)
                    else:
                        # 负采样：关系替换
                        relation_neg = random.randint(0, GlobalValue.relation_num - 1)
                        relation_neg = self.random_relation(pos, relation_neg)
                        self.loss = Gradient.train_kb(self.fb_h[pos], self.fb_l[pos], self.fb_r[pos], self.fb_h[pos],
                                                      self.fb_l[pos], relation_neg, self.loss)
                        # 归一化更新的负关系嵌入
                        Utils.norm(GlobalValue.relation_vec[relation_neg], GlobalValue.vector_len)
                    self.update_relation(pos)  # 更新关系向量
                    # 进行归一化更新完后的向量
                    Utils.norm(GlobalValue.relation_vec[self.fb_r[pos]], GlobalValue.vector_len)
                    Utils.norm(GlobalValue.entity_vec[self.fb_h[pos]], GlobalValue.vector_len)
                    Utils.norm(GlobalValue.entity_vec[self.fb_l[pos]], GlobalValue.vector_len)
            print("epoch: %s %s" % (epoch, self.loss))
        self.Write_Vec2File("resource_classification/result/1_APT-C-06/relation2vec.txt", GlobalValue.relation_vec, GlobalValue.relation_num)
        self.Write_Vec2File("resource_classification/result/1_APT-C-06/entity2vec.txt", GlobalValue.entity_vec, GlobalValue.entity_num)

    def add(self, head: int, relation: int, tail: int, path2prob_list: List[Tuple[List[int], float]]) -> None:
        '''
        添加三元组和路径概率对
        '''
        self.fb_h.append(head)
        self.fb_r.append(relation)
        self.fb_l.append(tail)
        self.fb_path2prob.append(path2prob_list)

        key = (head, relation)
        if key not in self.head_relation2tail:
            self.head_relation2tail[key] = set()
        tail_set = self.head_relation2tail[key]
        tail_set.add(tail)

    def run(self, nepoch: int, nbatches: int) -> None:
        GlobalValue.relation_vec = [[Utils.uniform(-6 / Utils.sqrt(GlobalValue.vector_len), 6 / Utils.sqrt(GlobalValue.vector_len)) for j in range(GlobalValue.vector_len)] for i in range(GlobalValue.relation_num)]

        GlobalValue.entity_vec = [[Utils.uniform(-6 / Utils.sqrt(GlobalValue.vector_len), 6 / Utils.sqrt(GlobalValue.vector_len)) for j in range(GlobalValue.vector_len)] for i in range(GlobalValue.entity_num)]
        # relation_vec是关系向量矩阵, entity_vec是实体向量矩阵, 初始化为一个[-6/sqrt(vector_len), 6/sqrt(vector_len)]之间的随机数

        for i in range(GlobalValue.entity_num):
            Utils.norm(GlobalValue.entity_vec[i], GlobalValue.vector_len)
            # 将每一行实体向量归一化

        self.bfgs(nepoch, nbatches)
        # 进行BFGS优化算法
