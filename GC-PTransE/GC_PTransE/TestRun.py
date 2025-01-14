import io
from typing import Dict
# from GC_PTransE.Test import Test
from GC_PTransE.Test import Test
# from GC_PTransE.GlobalValue import GlobalValue
from GC_PTransE.GlobalValue import GlobalValue



class TestRun:
    def __init__(self):
        self.test = None

    def Read_Data(self, file_name: str, data2id: Dict[str, int], id2data: Dict[int, str]) -> int:
        '''
        读取数据文件并存储到字典中
        '''
        count = 0
        with io.open(file_name, mode="r", encoding="UTF-8") as f:
            for line in f:
                split_data = line.strip().split("\t")
                data2id[split_data[0]] = int(split_data[1])
                id2data[int(split_data[1])] = split_data[0]
                count += 1
        return count

    def GlobalValueInit(self) -> None:
        '''
        初始化全局变量
        '''
        GlobalValue.relation2id = {}
        GlobalValue.entity2id = {}
        GlobalValue.id2relation = {}
        GlobalValue.id2entity = {}
        GlobalValue.left_entity = {}
        GlobalValue.right_entity = {}
        GlobalValue.left_num = {}
        GlobalValue.right_num = {}


    def vec_add_value(self, entity_map: Dict[int, Dict[int, int]], key: int, value_k: int) -> None:
        '''
        统计某个key对应的value_k值的个数     key:关系，value_k：头实体/尾实体
        存入entity_map字典，              格式：{关系：{实体：该关系下对应的该实体数量，实体：数量....},...}
        entity_map为 left_entity或者right_entity
        '''
        if key not in entity_map:
            entity_map[key] = {}
        # 这一行代码通过将 entity_map[key] 赋值给 entity_value，
        # 建立了对于 entity_map 字典中某个 key 对应值的引用，后续直接对变量entity_value进行操作即可以操作entity_map[key]的值
        entity_value = entity_map[key]
        if value_k not in entity_value:
            entity_value[value_k] = 0
        entity_value[value_k] += 1

    def prepare(self) -> None:
        '''
        读取数据和数据处理
        '''
        self.GlobalValueInit()
        # 测试集中的实体、关系数据处理
        GlobalValue.entity_num = self.Read_Data("resource_classification/data/1_APT-C-06/1_APT-C-06_entity2id.txt", GlobalValue.entity2id, GlobalValue.id2entity)
        GlobalValue.relation_num = self.Read_Data("resource_classification/data/1_APT-C-06/1_APT-C-06_relation2id.txt", GlobalValue.relation2id, GlobalValue.id2relation)

        with io.open("resource_classification/data/1_APT-C-06/1_APT-C-06_test.txt", mode="r", encoding="UTF-8") as f:
            for line in f:
                split_data = line.strip().split("\t")
                head, tail, relation = split_data[0], split_data[1], split_data[2]
                if head not in GlobalValue.entity2id:
                    print(f"miss entity: {head}")
                if tail not in GlobalValue.entity2id:
                    print(f"miss entity: {tail}")
                if relation not in GlobalValue.relation2id:
                    GlobalValue.relation2id[relation] = GlobalValue.relation_num
                    GlobalValue.relation_num += 1

                # left_entity，right_entity：测试集中关系分别和头实体、尾实体的对应关系    格式：{关系：{实体：该关系下对应的该实体数量，实体：数量....},...}
                self.vec_add_value(GlobalValue.left_entity, GlobalValue.relation2id[relation], GlobalValue.entity2id[head])  # 将对应的关系和头实体转换为他们的序号，带入参数
                self.vec_add_value(GlobalValue.right_entity, GlobalValue.relation2id[relation], GlobalValue.entity2id[tail])  # 关系、尾实体对应的序号，带入参数

                # 因为add函数传进来的参数是false，所以只执行add函数中将测试集每个三元组中的头、关系、尾实体分别添加到fb_h(头)、fb_r（关系）、fb_l（尾）
                self.test.add(GlobalValue.entity2id[head], GlobalValue.relation2id[relation], GlobalValue.entity2id[tail], False)  # 头、关系、尾对应的序号做参数

        with io.open("resource_classification/data/1_APT-C-06/1_APT-C-06_train.txt", mode="r", encoding="UTF-8") as f:
            for line in f:
                split_data = line.strip().split("\t")
                head, tail, relation = split_data[0], split_data[1], split_data[2]
                if head not in GlobalValue.entity2id:
                    print(f"miss entity: {head}")
                if tail not in GlobalValue.entity2id:
                    print(f"miss entity: {tail}")
                if relation not in GlobalValue.relation2id:
                    GlobalValue.relation2id[relation] = GlobalValue.relation_num
                    GlobalValue.relation_num += 1

                # 因为add函数传进来的参数是false，所以只执行add函数中,,,格式：{{head_id,rel_id}:{tail_id},.........}
                self.test.add(GlobalValue.entity2id[head], GlobalValue.relation2id[relation], GlobalValue.entity2id[tail], True)

        print(f"entity number = {GlobalValue.entity_num}")
        print(f"relation number = {GlobalValue.relation_num}")

    def test_run(self) -> None:
        global test
        self.test = Test()
        self.prepare()
        self.test.run()
