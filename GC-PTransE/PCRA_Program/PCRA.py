import os
from collections import deque
from typing import List, Dict, Set, Deque
import numpy as np
from PCRA_Program import Utils

# 声明类型别名
Entity = str
Relation = int
Path = List[Relation]
PathStr = str

class PCRA:
    '''
    实现了基于路径的知识图谱推理（Path-based Reasoning Algorithm，PCRA）
    '''

    def __init__(self):
        self.relation2id = {}
        self.id2relation = {}
        #
        self.entity2id = {}
        self.id2entity = {}
        #
        self.relation_num = 0
        self.head_relation2tail = {}  # (头实体, 关系) -> (尾实体)
        self.head_tail2relation = {}  # (头实体, 尾实体) -> (关系)
        self.head_tail2path = {}  # (头实体, 尾实体) -> (关系路径)
        self.paths = {}  # 记录每条路径，以及该路径出现的次数
        self.path2relation = {}  # 记录每个路径推理的边("path->rel")，已经对应出现的次数
        self.path_valid = set()  # 存储符合条件的路径

    def init(self):
        self.relation2id = {}
        self.id2relation = {}
        #
        self.entity2id = {}
        self.id2entity = {}
        #
        self.head_relation2tail = {}
        self.head_tail2relation = {}
        self.head_tail2path = {}
        self.paths = {}
        self.path2relation = {}
        self.path_valid = set()

    # 读取文件  （参数：文件名，两个字典）   返回一个整数表示成功读取和处理的数据数量
    # 主要功能是从指定的文件中读取数据，并将数据加载到提供的两个字典中
    def read_data(self, file_name: str, data2id: Dict[str, int], id2data: Dict[int, str]) -> int:
        # count计数器：用于记录读取和处理的数据数量
        count = 0
        with open(file_name, encoding='utf-8') as f:
            for line in f:
                split_data = line.strip().split('\t') # 以‘\t’进行分割数据
                data2id[split_data[0]] = int(split_data[1])  # 将split_data[0]作为键，将int(split_data[1])作为值，将数据加载到data2id字典中，以实现数据到ID的映射
                id2data[int(split_data[1])] = split_data[0]  # 将int(split_data[1])作为键，将split_data[0]作为值，将数据加载到id2data字典中，以实现ID到数据的映射
                count += 1
        return count

    def prepare(self) -> None:
        # 读取了文件数据并返回文件数据数量  讲处理好的数据存在两个字典中
        # realtion2id：是relation到id的映射
        # id2relation：是id到relation的映射
        self.relation_num = self.read_data('resource_classification/data/1_APT-C-06/1_APT-C-06_relation2id.txt', self.relation2id, self.id2relation)
        self.entity_num = self.read_data('resource_classification/data/1_APT-C-06/1_APT-C-06_entity2id.txt', self.entity2id, self.id2entity)

        with open('resource_classification/data/1_APT-C-06/1_APT-C-06_train.txt', encoding='utf-8') as f:
            for line in f:
                split_data = line.strip().split('\t')
                head = split_data[0]  # 头实体
                tail = split_data[1]  # 尾实体
                relation_id = self.relation2id[split_data[2]]  # 关系通过字典映射为编号
                # 头实体-尾实体-关系
                Utils.map_add_relation(self.head_tail2relation, head, relation_id, tail)  # ?????
                # 头实体-关系-尾实体
                Utils.map_add_tail(self.head_relation2tail, head, relation_id, tail)

    # 新增方法来获取id2relation字典
    def get_id2entity(self):
        return self.id2entity

    def run(self) -> None:
        self.init()
        self.prepare()
        for head, visit_relation_tail in self.head_relation2tail.items():
            cur_entity_list = deque([head])
            cur_relation_list = deque()
            self.dfs(cur_entity_list, visit_relation_tail, cur_relation_list, 0, 1, 1.0)   # ???????
            self.dfs(cur_entity_list, visit_relation_tail, cur_relation_list, 0, 2, 1.0)
            self.dfs(cur_entity_list, visit_relation_tail, cur_relation_list, 0, 3, 1.0)
            self.dfs(cur_entity_list, visit_relation_tail, cur_relation_list, 0, 4, 1.0)
        self.write_path()
        self.write_confident()
        self.calculate_prob("1_APT-C-06_train")
        self.calculate_prob("1_APT-C-06_test")

    # 深度遍历算法  用来统计走过的路径
    def dfs(self, entity_list: Deque[Entity], relation_tail: Dict[Relation, Set[Entity]],
            relation_list: Deque[Relation], depth: int, max_depth: int, prob: float) -> None:
        """
        entity_list: 已经访问过的实体列表，避免重复访问
        relation_tail: 需要访问的关系集合，来自entity_list中最后一个实体
        relation_list: 已经访问过的关系列表
        depth：表示当前DFS遍历的深度的整数
        max_depth：表示遍历的最大深度的整数
       """
        # 如果relation_tail为None且当前深度小于max_depth，函数返回并不进行进一步的计算
        if relation_tail is None and depth < max_depth:
            return
        """
        如果当前深度等于max_depth，意味着已达到最大深度。
        在这种情况下，函数生成路径，并更新各种数据结构以存储路径、关系和概率。
        """
        if depth == max_depth:
            # 生成路径和路径出现次数
            head = entity_list[0]
            tail = entity_list[-1]   # 取列表最后一个元素
            path = [str(rel_id) for rel_id in relation_list]
            path_str = ' '.join(path)
            # 函数 map_add_path 的作用是将指定字符串 path_str（当前路径） 添加到字典 self.paths 中，并记录它出现的次数
            Utils.map_add_path(self.paths, path_str)

            # 生成路径和关系之间对应关系和出现次数  是不是说以后出现该路径可以判断为其→对应的关系？？？暂时这样理解的
            head_tail = head + " " + tail
            if head_tail in self.head_tail2relation:
                relation_set = self.head_tail2relation[head_tail]
                Utils.add_path_2_relation(self.path2relation, path_str, relation_set)
                # 这里打印每次添加到head_tail2path字典中的数据
                # print(f"Added to head_tail2path: {head_tail}, {path_str}, {prob}")

            # 生成头尾实体对应路径和概率
            Utils.map_add_relation_path(self.head_tail2path, head, tail, path_str, prob)
            return

        for relation_id, tail_set in relation_tail.items():
            cur_prob = prob * (1.0 / len(tail_set))
            relation_list.append(relation_id)
            for tail in tail_set:
                if tail not in entity_list:  # if尾实体没在当前实体队列中
                    entity_list.append(tail)  # 就将他添加进去
                    visit_relation_tail = self.head_relation2tail.get(tail, {})  # 获取尾实体的后续节点集合
                    # 进行以“吕布”为头实体继续进行dfs的递归，知道深度达到设定的最大深度
                    self.dfs(entity_list, visit_relation_tail, relation_list, depth + 1, max_depth, cur_prob)
                    entity_list.pop()
            relation_list.pop()

    # 新增方法来获取head_tail2path字典
    def get_head_tail2path(self):
        # 例如，打印所有头尾实体对及其对应的路径
        # for head_tail, paths in self.head_tail2path.items():
        #     print(f"Entity Pair: {head_tail}, Paths: {paths}")
        return self.head_tail2path

    # def dfs(self, entity_list: Deque[Entity], relation_tail: Dict[Relation, Set[Entity]],
    #         relation_list: Deque[Relation], depth: int, max_depth: int, prob: float) -> None:
    #     """
    #     entity_list: 已经访问过的实体列表，避免重复访问
    #     relation_tail: 需要访问的关系集合，来自entity_list中最后一个实体
    #     relation_list: 已经访问过的关系列表
    #     """
    #     if relation_tail is None and depth < max_depth:
    #         return
    #     if depth == max_depth:
    #         # 生成路径和路径出现次数
    #         head = entity_list[0]
    #         tail = entity_list[-1]
    #         path = [str(rel_id) for rel_id in relation_list]
    #         path_str = ' '.join(path)
    #         Utils.map_add_path(self.paths, path_str)
    #
    #         # 生成路径和关系之间对应关系和出现次数
    #         head_tail = head + " " + tail
    #         if head_tail in self.head_tail2relation:
    #             relation_set = self.head_tail2relation[head_tail]
    #             Utils.add_path_2_relation(self.path2relation, path_str, relation_set)
    #
    #         # 生成头尾实体对应路径和概率
    #         Utils.map_add_relation_path(self.head_tail2path, head, tail, path_str, prob)
    #         return
    #
    #     for relation_id, tail_set in relation_tail.items():
    #         probabilities = self.generate_random_probabilities(len(tail_set))
    #         relation_list.append(relation_id)
    #         for index, tail in enumerate(tail_set):
    #             cur_prob = prob * probabilities[index]
    #             if tail not in entity_list:
    #                 entity_list.append(tail)
    #                 visit_relation_tail = self.head_relation2tail.get(tail, {})
    #                 self.dfs(entity_list, visit_relation_tail, relation_list, depth + 1, max_depth, cur_prob)
    #                 entity_list.pop()
    #         relation_list.pop()
    #
    # def generate_random_probabilities(self, n):
    #     # 生成n个随机数
    #     random_numbers = np.random.dirichlet(np.ones(n), size=1)
    #     return random_numbers[0]
    # '''
    # 函数功能：生成n个随机数，且n个随机数的加和为1
    # 如果n=1,输出的随机概率值为1.0。这是因为在 Dirichlet 分布中，参数向量都是正数，而且其总和为1，
    # 所以生成的概率向量中的元素也将是1，满足了概率的定义。
    # 例子：n=3
    # random_probs = generate_random_probabilities(3)
    # print(random_probs)
    # 输出结果：[0.42852303 0.34927694 0.22220003]
    # '''


    def write_path(self):
        path = 'resource_classification/path_data/1_APT-C-06/path.txt'
        dir = os.path.dirname(path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        with open(path, 'w', encoding="UTF-8") as f:
            for head in self.head_relation2tail.keys():
                for tail in self.head_relation2tail.keys():
                    if head == tail:
                        continue
                    head_tail = head + " " + tail
                    if head_tail in self.head_tail2path:
                        path_prob_valid = Utils.generate_valid_path(self.head_tail2path, head_tail)
                        f.write(head_tail + "\n")  # 将头尾实体写入并换行  head tail 换行
                        f.write(str(len(path_prob_valid)))  # 字典的长度:path的个数
                        for path, prob in path_prob_valid.items():
                            self.path_valid.add(path)
                            split_path = path.split(" ")
                            f.write(" " + str(len(split_path)) + " " + path + " " + str(prob)) # path的长度，path的内容，path的概率
                        f.write("\n")
                        f.flush()

    def write_confident(self) -> None:
        with open('resource_classification/path_data/1_APT-C-06/confident.txt', 'w', encoding='utf-8') as f:
            for path in self.path_valid:
                out_list = []
                for i in range(self.relation_num):
                    tmp_path2relation = f"{path}->{i}"
                    if tmp_path2relation in self.path2relation:
                        prob = 1.0 / self.paths[path]  # prob：1/path出现的次数    概率越小，出现次数越多
                        #  f-string形成一个字符串， 用大括{ }表示被替换字段，其中直接填入替换内容即可。
                        str_ = f" {i} {self.path2relation[tmp_path2relation] * prob}"
                        out_list.append(str_)
                if out_list:
                    f.write(f"{len(path.split())} {path}\n") # path的长度 path的内容 换行
                    f.write(str(len(out_list))) # path可推导的head_relation的个数
                    for out in out_list:
                        f.write(out) # head_relation的内容 path推head_relation的概率：推出head_relation的次数*path出现的概率
                    f.write('\n')
        return


    def calculate_prob(self, file_name: str) -> None:
        with open(f'resource_classification/data/1_APT-C-06/{file_name}.txt', encoding='utf-8') as f:  # 文件名：train、test
            with open(f'resource_classification/path_data/1_APT-C-06/{file_name}_prob.txt', 'w', encoding='utf-8') as writer: # 新打开的train_prob文件
                for line in f:   # 处理train和test文件
                    split_data = line.strip().split('\t')
                    head = split_data[0]
                    tail = split_data[1]
                    relation_id = self.relation2id[split_data[2]]

                    head_tail = head + " " + tail
                    path_prob_valid = {}
                    if head_tail in self.head_tail2path:
                        path_prob_valid = Utils.generate_valid_path(self.head_tail2path, head_tail)

                    writer.write(f"{head}\t{tail}\t{relation_id}\n")
                    writer.write(str(len(path_prob_valid)))
                    for path, prob in path_prob_valid.items():
                        writer.write(f" {len(path.split())} {path} {prob}")
                    writer.write('\n')


