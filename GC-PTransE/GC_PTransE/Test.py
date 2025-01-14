import io
from typing import List, Dict, Set, Tuple
from GC_PTransE.GlobalValue import GlobalValue
from GC_PTransE.Gradient import Gradient
from PCRA_Program.PCRA import PCRA


class Test:
    def __init__(self):
        self.fb_h: List[int] = []
        self.fb_l: List[int] = []
        self.fb_r: List[int] = []
        self.head_relation2tail: Dict[Tuple[int, int], Set[int]] = {}

    def add(self, head: int, relation: int, tail: int, flag: bool) -> None:
        '''
          说的不对： head_relation2tail用于存放正确的三元组
                    flag=True 表示该三元组关系正确

          flag只是表示应该执行哪段代码
        '''
        if flag:
            key = (head, relation)
            if key not in self.head_relation2tail:
                self.head_relation2tail[key] = set()
            tail_set = self.head_relation2tail[key]  # head_relation2tail字典中存放：{{头1，关系1}：{尾1}，{头2，关系2}：{尾2}....}
            tail_set.add(tail)
        else:
            self.fb_h.append(head)
            self.fb_l.append(tail)
            self.fb_r.append(relation)

    def Read_Vec_File(self, file_name: str, vec: List[List[float]]) -> None:
        '''
        读取文件中的向量
        '''
        with io.open(file_name, mode="r", encoding="UTF-8") as f:
            # 逐行读取文件
            for i, line in enumerate(f):
                # 去除首尾空白并按制表符分割行，得到向量的字符串表示
                line_split = line.strip().split("\t")
                # 遍历每个元素的索引
                for j in range(GlobalValue.vector_len):
                    # 将字符串转换为浮点数，并将其存储到传递的二维列表 vec 中的相应位置
                    vec[i][j] = float(line_split[j])

    def relation_add(self, relation_num: Dict[int, int], relation: int) -> None:
        '''
        计算每个关系出现的次数

        关系不在relation_num字典中则置为0，并加1
        在relation_num中直接加1
        '''
        if relation not in relation_num:
            relation_num[relation] = 0
        count = relation_num[relation]
        relation_num[relation] = count + 1

    def map_add_value(self, tmp_map: Dict[int, int], id: int, value: int) -> None:
        '''
        统计字典某个key对应的value之和
        '''
        if id not in tmp_map:
            tmp_map[id] = 0
        tmp_value = tmp_map[id]
        tmp_map[id] = tmp_value + value

    def hrt_isvalid(self, head: int, relation: int, tail: int) -> bool:
        '''
        如果实体之间已经存在训练集中关系，则不需要计算距离
        如果头实体与尾实体一致，也排除该关系的距离计算
        '''
        if head == tail:
            return True
        key = (head, relation)
        # values是得到head_relation2tail中头、关系  对应的尾实体
        values = self.head_relation2tail.get(key)
        '''
        not values: 检查 values 是否为空或不存在（如果 values 为空，或者是一个假值，例如空列表、空字符串等，那么条件成立）。
        tail not in values: 检查 tail 是否不在 values 中
        
        如果上述两个条件中的任意一个成立（即 values 为空或 tail 不在 values 中），那么整个条件表达式为真，执行 return False。
        否则，如果两个条件都不成立，就执行 return True。
        '''
        if not values or tail not in values:
            return False
        else:
            return True


    # 分别计算实体和关系的MR和Hits@10
    def run(self) -> None:
        '''
        这行代码创建了一个名为 relation_vec 的二维列表，其中包含 GlobalValue.relation_num 行，每行包含 GlobalValue.vector_len 个元素。
        所有元素都被初始化为浮点数 0.0。这样的结构通常用于存储向量，每一行表示一个关系对应的向量。
        eg: # 示例参数
            GlobalValue.relation_num = 3
            GlobalValue.entity_num = 4
            GlobalValue.vector_len = 2

            # 初始化关系向量
            GlobalValue.relation_vec = [[0.0] * GlobalValue.vector_len for _ in range(GlobalValue.relation_num)]
            # 结果：[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]

            # 初始化实体向量
            GlobalValue.entity_vec = [[0.0] * GlobalValue.vector_len for _ in range(GlobalValue.entity_num)]
            # 结果：[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]
        '''
        GlobalValue.relation_vec = [[0.0] * GlobalValue.vector_len for _ in range(GlobalValue.relation_num)]
        GlobalValue.entity_vec = [[0.0] * GlobalValue.vector_len for _ in range(GlobalValue.entity_num)]

        # Read_Vec_File：读取文件中的向量，将其存储到传递的二维列表 relaion_vec/entity_vec 中的相应位置
        self.Read_Vec_File("resource_classification/result/1_APT-C-06/relation2vec.txt", GlobalValue.relation_vec)
        self.Read_Vec_File("resource_classification/result/1_APT-C-06/entity2vec.txt", GlobalValue.entity_vec)
        # 控制台打印
        # print(GlobalValue.relation_vec)
        # print(GlobalValue.entity_vec)

        lsum, rsum, msum, m_sum = 0, 0, 0, 0
        lp_n, rp_n, mp_n, m_p_n= 0, 0, 0, 0
        head_hits_1, head_hits_5, head_hits_10 = 0, 0, 0
        tail_hits_1, tail_hits_5, tail_hits_10 = 0, 0, 0
        total_hits_1, total_hits_5, total_hits_10 = 0, 0, 0
        lsum_r, rsum_r, msum_r = dict(), dict(), dict()
        lp_n_r, rp_n_r, mp_n_r = dict(), dict(), dict()
        rel_num = dict()

        with io.open("resource_classification/result/1_APT-C-06/output_detail.txt", mode="w", encoding="UTF-8") as writer:
            print(f"Total iterations = {len(self.fb_l)}")
            for idx in range(len(self.fb_l)):
                print('测试集第' + (str(idx + 1)) + '个三元组')

                # 选择测试集中的一个三元组  按顺序来  idx=0,1,2,3....31
                head, tail, relation = self.fb_h[idx], self.fb_l[idx], self.fb_r[idx]
                # 将当前三元组关系以及出现次数放入rel_num字典中
                self.relation_add(rel_num, relation)

                # 创建 PCRA 实例
                pcra_instance = PCRA()
                # 初始化并准备数据
                pcra_instance.init()
                pcra_instance.prepare()
                pcra_instance.run()
                # 获取id2entity字典
                id_entity_mapping = pcra_instance.get_id2entity()
                head_tail2path_mapping = pcra_instance.get_head_tail2path()
                # 假设我们已经有了头实体和尾实体
                h = id_entity_mapping[head]
                t = id_entity_mapping[tail]
                h_t = h + ' ' + t

                sum_confi_pvec = 0.0  # 给变量一个初始值
                sum_prob = 0.0

                if h_t in head_tail2path_mapping:
                    writer.write('测试集第' + (str(idx + 1)) + '个三元组\n')
                    # 对三元组头实体进行替换
                    head_dist = []  # 记录替换头实体之后的三元组以及其能量打分
                    for e in range(GlobalValue.entity_num):  # 实体数量89
                        '''
                            hrt_isvalid函数：
                            如果实体之间已经存在训练集中关系，则不需要计算距离
                            如果头实体与尾实体一致，也排除该关系的距离计算
                        '''
                        if self.hrt_isvalid(e, relation, tail):
                            continue
                        h_random = id_entity_mapping[e]
                        h_random_t = h_random + ' ' + t
                        if h_random_t in head_tail2path_mapping:
                            sum_ = Gradient.calc_energy(e, tail, relation, h_random_t, head_tail2path_mapping, sum_confi_pvec, sum_prob)
                        else:
                            continue

                        head_dist.append((e, sum_))  # 序号（头实体id），sum值
                    head_dist.sort(key=lambda x: x[1])  # 按照列表第二个关键字（得分）进行升序排序  即按照sum_值进行排序
                    # print(len(head_dist))
                    writer.write('头实体预测：\n')
                    for i in range(len(head_dist)):
                        cur_head = head_dist[i][0]
                        if cur_head == head:  # 当找到与测试集中三元组一样的头实体后，写入并跳出循环
                            lsum += i + 1
                            m_sum += i + 1
                            self.map_add_value(lsum_r, relation, i)
                            # 计算hit@10的变量
                            if i <= 10:  # i为前10时
                                lp_n += 1  # lp_n置为1
                                m_p_n += 1
                                self.map_add_value(lp_n_r, relation, 1)
                            h_t_correct = h + ' ' + t
                            str_ = f"{GlobalValue.id2entity[head]}\t{GlobalValue.id2relation[relation]}\t{GlobalValue.id2entity[tail]}, dist={Gradient.calc_energy(head, tail, relation, h_t_correct, head_tail2path_mapping, sum_confi_pvec, sum_prob)}, {i}\n\n"
                            writer.write(str_)
                            writer.flush()
                            break
                        else:
                            head_en = id_entity_mapping[cur_head]
                            h_t_error = head_en + ' ' + t
                            temp_str = f"{GlobalValue.id2entity[cur_head]}\t{GlobalValue.id2relation[relation]}\t{GlobalValue.id2entity[tail]}, dist={Gradient.calc_energy(cur_head, tail, relation, h_t_error, head_tail2path_mapping, sum_confi_pvec, sum_prob)}, {i}\n"
                            writer.write(temp_str)
                            writer.flush()

                    tail_dist = []
                    for e in range(GlobalValue.entity_num):
                        if self.hrt_isvalid(head, relation, e):
                            continue
                        t_random = id_entity_mapping[e]
                        h_t_random = h + ' ' + t_random
                        if h_t_random in head_tail2path_mapping:
                            sum_ = Gradient.calc_energy(head, e, relation, h_t_random, head_tail2path_mapping, sum_confi_pvec, sum_prob)
                        else:
                            continue

                        tail_dist.append((e, sum_))
                    tail_dist.sort(key=lambda x: x[1])  # 按照列表第二个关键字进行升序排序  sum值进行排序
                    writer.write('尾实体预测：\n')
                    for i in range(len(tail_dist)):
                        cur_tail = tail_dist[i][0]
                        if cur_tail == tail:
                            rsum += i + 1
                            m_sum += i + 1
                            self.map_add_value(rsum_r, relation, i)
                            # 计算hit@10的变量
                            if i <= 10:
                                rp_n += 1
                                m_p_n += 1
                                self.map_add_value(rp_n_r, relation, 1)
                            h_t_correct = h + ' ' + t
                            str_ = f"{GlobalValue.id2entity[head]}\t{GlobalValue.id2relation[relation]}\t{GlobalValue.id2entity[tail]}, dist={Gradient.calc_energy(head, tail, relation, h_t_correct, head_tail2path_mapping, sum_confi_pvec, sum_prob)}, {i}\n\n"
                            writer.write(str_)
                            writer.flush()
                            break
                        else:
                            tail_en = id_entity_mapping[cur_tail]
                            h_t_error = h + ' ' + tail_en
                            temp_str = f"{GlobalValue.id2entity[head]}\t{GlobalValue.id2relation[relation]}\t{GlobalValue.id2entity[cur_tail]}, dist={Gradient.calc_energy(head, cur_tail, relation, h_t_error, head_tail2path_mapping, sum_confi_pvec, sum_prob)}, {i}\n"
                            writer.write(temp_str)
                            writer.flush()
                            # continue

                    relation_dist = []
                    for r in range(GlobalValue.relation_num):
                        if self.hrt_isvalid(head, r, tail):
                            continue
                        h_t = h + ' ' + t
                        if h_t in head_tail2path_mapping:
                            sum_ = Gradient.calc_energy(head, tail, r, h_t, head_tail2path_mapping,
                                                        sum_confi_pvec, sum_prob)
                        else:
                            continue

                        relation_dist.append((r, sum_))
                    relation_dist.sort(key=lambda x: x[1])  # 按照列表第二个关键字进行升序排序  sum值进行排序
                    writer.write('关系预测：\n')
                    for i in range(len(relation_dist)):
                        cur_relation = relation_dist[i][0]
                        if cur_relation == relation:
                            msum += i + 1   # msum是原本关系排名计算
                            m_sum += i + 1  #总
                            self.map_add_value(msum_r, relation, i)
                            # 计算hit@10的变量
                            if i <= 10:
                                mp_n += 1   # mp_n是原本关系排名在前10的累计和计算  关系自己的
                                m_p_n += 1   #总
                                self.map_add_value(mp_n_r, relation, 1)
                            h_t_correct = h + ' ' + t
                            str_ = f"{GlobalValue.id2entity[head]}\t{GlobalValue.id2relation[relation]}\t{GlobalValue.id2entity[tail]}, dist={Gradient.calc_energy(head, tail, relation, h_t, head_tail2path_mapping, sum_confi_pvec, sum_prob)}, {i}\n\n"
                            writer.write(str_)
                            writer.flush()
                            break
                        else:
                            temp_str = f"{GlobalValue.id2entity[head]}\t{GlobalValue.id2relation[cur_relation]}\t{GlobalValue.id2entity[tail]}, dist={Gradient.calc_energy(head, tail, cur_relation, h_t, head_tail2path_mapping, sum_confi_pvec, sum_prob)}, {i}\n"
                            writer.write(temp_str)
                            writer.flush()
                            # continue

        print(f"m_sum = {m_sum}, m_p_n={m_p_n},tail number = {len(self.fb_l)}")

        print(f"entity left: {(lsum * 1.0) / len(self.fb_l)}, {(lp_n * 1.0) / len(self.fb_l)}")
        print(f"entity right: {(rsum * 1.0) / len(self.fb_h)}, {(rp_n * 1.0) / len(self.fb_h)}")
        print(f"mid: {(msum * 1.0) / len(self.fb_h)}, {(mp_n * 1.0) / len(self.fb_h)}")

        print(f"total : {(m_sum * 1.0) /(2 * len(self.fb_l))}, {(m_p_n * 1.0) / (2 * len(self.fb_l))}")

