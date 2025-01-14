from typing import List
import random
import math

class Utils:
    random = random.Random()
    PI = math.pi

    @staticmethod
    def sqrt(x: float) -> float:
        return math.sqrt(x)

    @staticmethod
    def sqr(x: float) -> float:
        return x * x

    @staticmethod
    def abs(x: float) -> float:
        return abs(x)

    @staticmethod
    def exp(x: float) -> float:
        return math.exp(x)

    @staticmethod
    def normal(x: float) -> float:
        # 标准高斯分布
        return Utils.exp(-0.5 * Utils.sqr(x)) / Utils.sqrt(2 * Utils.PI)

    @staticmethod
    def rand() -> int:
        return Utils.random.randint(0, 32767)

    @staticmethod
    def uniform(min_: float, max_: float) -> float:
        # 生成一个在[min, max)范围内的浮点数，类似于Python中的uniform函数
        return min_ + (max_ - min_) * Utils.random.random()

    @staticmethod
    # 计算输入向量 a 的长度（L2范数或模），也就是向量中各个维度上元素的平方和的平方根
    def vec_len(a: List[float], vec_size: int) -> float:
        # 计算向量的长度
        res = 0.0
        for i in range(vec_size):
            res += Utils.sqr(a[i])
        return Utils.sqrt(res)  # 相当于：根号下a²+b²+c²+....，各个维度的平方加和开平方

    @staticmethod
    # 将输入的列表 a 中的元素限制在 1 以下，以确保向量的长度（L2范数或模）不超过1
    # 归一化
    def norm(a: List[float], vec_size: int) -> None:
        # 将元素a限制在1以下
        x = Utils.vec_len(a, vec_size)  # 计算出向量的长度(模长)
       # 函数检查 x 是否大于1。如果 x 大于1，说明向量的长度超过了1，需要进行规范化。
       # 如果 x 大于1，则进入一个循环，该循环遍历向量 a 的每个维度（通过 vec_size 指定）
        if x > 1:
            for i in range(vec_size):
                a[i] /= x   # 使每个向量除以它的长度

    @staticmethod
    def rand_max(x: int) -> int:
        # 在(0, x)范围内生成一个随机数
        res = (Utils.rand() * Utils.rand()) % x
        while res < 0:
            res += x
        return res
