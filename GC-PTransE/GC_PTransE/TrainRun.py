from GC_PTransE.Train import Train
from GC_PTransE.GlobalValue import GlobalValue

def read_data(file_name, data2id, id2data):
    count = 0
    with open(file_name, encoding='UTF-8') as f:
        for line in f:
            split_data = line.strip().split('\t')
            data2id[split_data[0]] = int(split_data[1])  # 实体名  编号
            id2data[int(split_data[1])] = split_data[0]  # 编号  实体名
            count += 1  # 计数器
    return count

train = Train()  # 实例化训练对象
class TrainRun:

    def train_run(self):
        # 初始化参数
        # nepoch = 1200  # 迭代次数(原始)
        # nbatches = 150  # 数据块数量

        # epochs_values = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        # batches_values = [50, 100, 150, 200, 250, 300]

        # for nepoch in epochs_values:
        #     for nbatches in batches_values:
        nepoch = 450  # 迭代次数
        nbatches = 50  # 数据块数量
        print("iteration times =", nepoch)
        print("nbatches =", nbatches)
        self.prepare()  # 准备数据
        # 训练模型
        train.run(nepoch, nbatches)

    def prepare(self):   # 准备训练数据
        self.global_value_init()
        # 初始化全局变量  处理实体和关系数据，保存并统计个数
        GlobalValue.entity_num = read_data("resource_classification/data/1_APT-C-06/1_APT-C-06_entity2id.txt", GlobalValue.entity2id, GlobalValue.id2entity)  # 读取实体信息
        GlobalValue.relation_num = read_data("resource_classification/data/1_APT-C-06/1_APT-C-06_relation2id.txt", GlobalValue.relation2id, GlobalValue.id2relation)  # 读取关系信息

        # 读取训练数据
        f = open("resource_classification/path_data/1_APT-C-06/1_APT-C-06_train_prob.txt", encoding='UTF-8')
        for line in f:
            split_data = line.strip().split('\t')
            # split_data = line.strip().split(' ')
            head_id = GlobalValue.entity2id[split_data[0]]  # 头实体ID
            tail_id = GlobalValue.entity2id[split_data[1]]  # 尾实体ID
            relation_id = int(split_data[2])  # 关系ID

            # 读取路径信息  保存路径和概率  [(路径，概率)]
            path_info = f.readline().strip().split(' ')
            path2prob_list = []
            i = 1
            while i < len(path_info):
                path_length = int(path_info[i])  # 路径长度
                relation_id_list = []
                for j in range(1, path_length + 1):
                    relation_id_list.append(int(path_info[i + j]))  # 路径中每个关系的ID
                prob = float(path_info[i + path_length + 1])  # 路径对应的概率
                path2prob = (relation_id_list, prob)
                path2prob_list.append(path2prob)  # 路径与概率的二元组  [(路径，概率)]

                i += path_length + 2
            train.add(head_id, relation_id, tail_id, path2prob_list)

        f.close()

        path_confidence = {}  # 初始化路径置信度
        # 读取路径置信度
        f = open("resource_classification/path_data/1_APT-C-06/confident.txt", encoding='UTF-8')
        for line in f:
            line_split = line.strip().split(' ')
            path_str = ' '.join(line_split[1:])
            path_info = f.readline().strip().split(' ')
            i = 1
            while i < len(path_info):
                relation_id = int(path_info[i])  # 关系ID
                prob = float(path_info[i + 1])  # 路径置信度

                path2relation = (path_str, relation_id)
                path_confidence[path2relation] = prob

                i += 2
        f.close()

        # 输出实体和关系数量
        print("entity number =", GlobalValue.entity_num)
        print("relation number =", GlobalValue.relation_num)

    def global_value_init(self):
        GlobalValue.relation2id = {}
        GlobalValue.entity2id = {}
        GlobalValue.id2relation = {}
        GlobalValue.id2entity = {}
        GlobalValue.left_entity = {}
        GlobalValue.right_entity = {}
        GlobalValue.left_num = {}
        GlobalValue.right_num = {}
        GlobalValue.path_confidence = {}
        pass

