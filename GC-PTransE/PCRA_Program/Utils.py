from typing import Dict, Set

def map_add_HeadTail(map: Set[str], head: str, tail: str) -> None:
    """
    添加以 head 和 tail 为起止点的字符串到 set 中。

    Args:
        map (Set[str]): 存放字符串的 set 对象
        head (str): 起点字符串
        tail (str): 终点字符串
    """
    key = head + " " + tail
    map.add(key)

def map_add_path(map: Dict[str, int], path: str) -> None:
    """
    将指定字符串 path 加入到字典 map 中。如果 path 不存在则创建一个新的键值对。

    它将指定的字符串 path 添加到字典 map 中，并记录出现的次数。
    如果 path 不存在于字典中，则创建一个新的键值对，并将值初始化为 1。
    如果 path 已经存在于字典中，则将对应的值加 1。

    Args:
        map (Dict[str, int]): 存放字符串到整数映射的字典
        path (str): 待添加的字符串
    """
    map[path] = map.get(path, 0) + 1

def map_add_tail(map: Dict[str, Dict[int, Set[str]]], head: str, relation_id: int, tail: str) -> None:
    """
    将以 head 和 relation_id 为键的字典加入到以 head 为键的字典 map 中。如果 head 不存在则创建一个新的键值对。

    Args:
        map (Dict[str, Dict[int, Set[str]]]): 存放字符串和整数映射的字典
        head (str): 起点字符串
        relation_id (int): 表示关系的整数
        tail (str): 终点字符串
    """
    if head not in map:
        map[head] = {}
    relation2tail = map[head]
    if relation_id not in relation2tail:
        relation2tail[relation_id] = set()
    tail_set = relation2tail[relation_id]
    tail_set.add(tail)

def map_add_relation(map: Dict[str, Set[int]], head: str, relation_id: int, tail: str) -> None:
    """
    将以 head 和 tail 为键的 set 加入到以 head 和 tail 为键的字典 map 中。如果该键不存在则创建一个新的键值对。

    Args:
        map (Dict[str, Set[int]]): 存放字符串到整数集合的字典  ????
        head (str): 起点字符串
        relation_id (int): 表示关系的整数
        tail (str): 终点字符串
    """
    key = head + " " + tail
    if key not in map:
        map[key] = set()  # 从字典map中获取键key对应的值，即一个集合
    relation_set = map[key]
    relation_set.add(relation_id)

def map_add_relation_path(map: Dict[str, Dict[str, float]], head: str, tail: str, relation_path: str, prob: float) -> None:
    """
    将关系路径以及对应的概率加入到以 head 和 tail 为键的字典 map 中。如果该键不存在则创建一个新的键值对。

    Args:
        map (Dict[str, Dict[str, float]]): 存放字符串到字符串和浮点数映射的字典
        head (str): 起点字符串
        tail (str): 终点字符串
        relation_path (str): 表示关系路径的字符串
        prob (float): 表示路径概率的浮点数
    """
    head_tail = head + " " + tail
    if head_tail not in map:
        map[head_tail] = {}
    path_set = map[head_tail]
    # 在path_set字典中添加或更新键relation_path的值。
    # 它使用get方法检索与relation_path关联的当前值，如果键不存在，则使用默认值0.0。
    # 然后，将prob的值加上检索到的值，并将结果重新赋值给path_set[relation_path]。
    path_set[relation_path] = path_set.get(relation_path, 0.0) + prob

def add_path_2_relation(path2relation_set: Dict[str, int], path: str, relation_set: Set[int]) -> None:
    """
    将 path 和 relation_set 中的每个元素构成的字符串加入到字典 path2relation_set 中。

    Args:
        path2relation_set (Dict[str, int]): 存放字符串到整数映射的字典
        path (str): 表示路径的字符串
        relation_set (Set[int]): 存放整数的集合
    """
    for relation in relation_set:
        path_relation = path + "->" + str(relation)
        map_add_path(path2relation_set, path_relation)

def generate_valid_path(head_tail2path: Dict[str, Dict[str, float]], head_tail: str) -> Dict[str, float]:
    """
    从以 head_tail 为键的字典中提取路径，并根据条件筛选出符合要求的路径。

    Args:
        head_tail2path (Dict[str, Dict[str, float]]): 存放字符串到字符串和浮点数映射的字典
        head_tail (str): 表示起止点的字符串

    Returns:
        Dict[str, float]: 存放字符串到浮点数映射的字典
    """
    path_prob = {} # 记录所有路径以及相应的路径概率
    path_prob_valid = {} # 记录符合概率阈值的路径

    sum = 0.0 # 用于归一化
    path_set = head_tail2path.get(head_tail, {})
    for path, prob in path_set.items():
        path_prob[path] = prob
        sum += prob

    for path, prob in path_prob.items():
        prob /= sum
        path_prob[path] = prob
        if prob > 0.01: # 筛选条件
            path_prob_valid[path] = prob

    return path_prob_valid
