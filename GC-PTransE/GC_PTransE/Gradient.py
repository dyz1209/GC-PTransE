from typing import List
from typing import Dict
from GC_PTransE.GlobalValue import GlobalValue
from GC_PTransE.Utils import Utils
# from GC_PTransE.Test import Test
from PCRA_Program.PCRA import PCRA


class Gradient:
    @staticmethod
    def calc_sum(e1: int, e2: int, rel: int) -> float:  # ||h+r-t||=>||t-h-r||  基础能量函数计算
        sum_ = 0.0
        # print(GlobalValue.vector_len)
        for i in range(GlobalValue.vector_len):
            sum_ += Utils.abs(GlobalValue.entity_vec[e2][i] - GlobalValue.entity_vec[e1][i] - GlobalValue.relation_vec[rel][i])
        return sum_

    @staticmethod
    def calc_sum_pvec(e1: int, e2: int, p_vec: List[float]) -> float:  # ||h+p-t||=>||t-h-p||  基础能量函数计算
        sum_ = 0.0
        # print(GlobalValue.vector_len)
        for i in range(GlobalValue.vector_len):
            # print(p_vec[i])
            sum_ += Utils.abs(
                GlobalValue.entity_vec[e2][i] - GlobalValue.entity_vec[e1][i] - p_vec[i])
        return sum_

    @staticmethod
    def calc_energy(head: int, tail: int, relation: int, h_t: str, head_tail2path_mapping:Dict[str, Dict[str, float]], sum_confi_pvec:float, sum_prob:float) -> float:  # ||h+r-t||=>||t-h-r||  基础能量函数计算
        paths_and_probs = head_tail2path_mapping[h_t]
        for key, value in paths_and_probs.items():
            # key = path_prob.keys()
            # prob = path_prob.values()

            p_vec = [0.0] * GlobalValue.vector_len

            paths = key.split()
            for j in range(len(paths)):
                # path_var_name = 'path_' + str(j + 1)
                path_var_value = int(paths[j])
                # p为关系向量加和，计算p
                for v in range(GlobalValue.vector_len):
                    p_vec[v] += GlobalValue.relation_vec[path_var_value][v]
            # 计算能量函数:||h+p-t||
            sum_pvec = Gradient.calc_sum_pvec(head, tail, p_vec)
            # print(sum_pvec)
            confi_pvec = value * sum_pvec  # 置信度*E(h,p,t)
            sum_confi_pvec += confi_pvec  # 多条路径p的置信度*E(h,p,t)求和
            sum_prob += value  # 多条路径p的置信度求和
            # print(sum_prob)
        E_hPt = sum_confi_pvec / sum_prob  # 计算E(h,P,t)  P是多个p,即多条路径

        # 计算能量函数:||h+r-t||
        E_hrt = Gradient.calc_sum(head, tail, relation)
        # 总的能量函数
        sum_ = E_hrt + E_hPt
        # print(sum_)

        return sum_


    @staticmethod
    def train_kb(head_a: int, tail_a: int, relation_a: int, head_b: int, tail_b: int, relation_b: int, res: float) -> float:
        # 计算两个实体之间的距离和关系之间的距离
        sum1 = Gradient.calc_sum(head_a, tail_a, relation_a)
        sum2 = Gradient.calc_sum(head_b, tail_b, relation_b)
        # 如果它们之间的距离远于边距，更新梯度
        if sum1 + GlobalValue.margin > sum2:
            res += GlobalValue.margin + sum1 - sum2
            Gradient.gradient(head_a, tail_a, relation_a, -1)
            Gradient.gradient(head_b, tail_b, relation_b, 1)
        return res

    @staticmethod
    def gradient(head: int, tail: int, relation: int, beta: int) -> None:
        # 更新关系嵌入和实体嵌入
        for i in range(GlobalValue.vector_len):
            delta = GlobalValue.entity_vec[tail][i] - GlobalValue.entity_vec[head][i] - GlobalValue.relation_vec[relation][i]
            x = 1 if delta > 0 else -1
            GlobalValue.relation_vec[relation][i] -= x * GlobalValue.learning_rate * beta
            GlobalValue.entity_vec[head][i] -= x * GlobalValue.learning_rate * beta
            GlobalValue.entity_vec[tail][i] += x * GlobalValue.learning_rate * beta

    '''
    train_path 方法
    
    输入参数：
    relation：当前关系的索引。
    neg_relation：负例关系的索引。
    path：路径列表，包含一系列关系的索引。
    alpha：一个权重因子，用于调整损失函数的贡献度。
    loss：当前的损失值。
    
    计算距离：
    使用calc_path方法计算当前关系与路径的距离（sum1），以及负例关系与路径的距离（sum2）。
    这个距离是通过比较关系向量和路径上的关系向量之间的差异来计算的。
    比较和更新损失：

    检查当前关系与路径的距离加上一个预设的边界（GlobalValue.margin_relation）是否大于负例关系与路径的距离。
    如果是，表明当前关系与路径的关联比负例关系更弱，需要通过调整向量来减少这种差异。
    更新损失函数值，将差异乘以alpha后加到loss上。
    梯度更新：

    调用gradient_path方法来更新关系向量。
    对于当前关系和负例关系，根据它们与路径的距离差异调整它们的嵌入向量。
    这种调整是为了使当前关系向量更接近路径，而使负例关系向量远离路径。
    返回更新的损失值：

    返回经过调整后的损失值。
    总结
    train_path函数是用于更新知识图谱中关系嵌入的关键方法，它根据关系与路径之间的距离来调整关系向量。
    这种方法的目的是减少模型对于正例（当前关系与路径之间的关系）的误差，同时增加对于负例（随机选择的关系与路径之间的关系）的区分度。
    通过这种方式，模型可以更好地学习和区分不同的关系，特别是在它们在路径上表现出来的时候。
    
    '''


    @staticmethod
    def train_path(relation: int, neg_relation: int, path: List[int], alpha: float, loss: float) -> float:
        # 计算路径之间的距离
        sum1 = Gradient.calc_path(relation, path)
        sum2 = Gradient.calc_path(neg_relation, path)
        # 如果它们之间的距离远离边距，更新梯度
        if sum1 + GlobalValue.margin_relation > sum2:
            loss += alpha * (sum1 + GlobalValue.margin_relation - sum2)
            Gradient.gradient_path(relation, path, -1 * alpha)
            Gradient.gradient_path(neg_relation, path, alpha)
        return loss

    @staticmethod
    def gradient_path(relation: int, path: List[int], beta: float) -> None:
        """
        相关联的路径和关系之间的空间位置相近，反之疏远
        """
        # 更新路径和关系之间的距离
        for i in range(GlobalValue.vector_len):
            x = GlobalValue.relation_vec[relation][i]
            for path_id in path:
                x -= GlobalValue.relation_vec[path_id][i]
            flag = 1 if x > 0 else -1
            GlobalValue.relation_vec[relation][i] += beta * GlobalValue.learning_rate * flag
            for path_id in path:
                GlobalValue.relation_vec[path_id][i] -= beta * GlobalValue.learning_rate * flag

    @staticmethod
    # 计算给定关系（relation）和路径（path）之间的一个数值（浮点数），它表示关系和路径之间的某种空间差异或距离
    # 用于衡量关系和路径之间的相似性或差异性，分析和评估关系的特征
    def calc_path(relation: int, path: List[int]) -> float:
        sum_ = 0.0
        for i in range(GlobalValue.vector_len):  # 向量的维数
            x = GlobalValue.relation_vec[relation][i]
            for path_id in path:
                x -= GlobalValue.relation_vec[path_id][i]
            sum_ += abs(x)   # abs求绝对值函数
        return sum_
