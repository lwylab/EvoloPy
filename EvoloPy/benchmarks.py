from typing import List, Union, Any

import numpy as np
from opfunu.cec_based import cec2022

# 类型别名
ArrayLike = Union[List[float], np.ndarray]

# 基准函数常量
ACKLEY_A = 20
ACKLEY_B = 0.2
ACKLEY_C = 2 * np.pi
RASTRIGIN_A = 10


class CEC2022Functions:
    """CEC 2022基准函数集合"""

    @staticmethod
    def _evaluate_cec2022(x: ArrayLike, func_num: int, dim: int = 10) -> float:
        """CEC 2022函数的通用评估方法

        参数:
            x: 输入向量
            func_num: 函数编号(1-12)
            dim: 问题维度

        返回:
            在x点的函数值
        """
        func_name = f'F{func_num}2022'
        return getattr(cec2022, func_name)(ndim=dim).evaluate(x)

    @staticmethod
    def F12022(x: ArrayLike) -> float:
        return CEC2022Functions._evaluate_cec2022(x, 1)

    @staticmethod
    def F22022(x: ArrayLike) -> float:
        return CEC2022Functions._evaluate_cec2022(x, 2)

    @staticmethod
    def F32022(x: ArrayLike) -> float:
        return CEC2022Functions._evaluate_cec2022(x, 3)

    @staticmethod
    def F42022(x: ArrayLike) -> float:
        return CEC2022Functions._evaluate_cec2022(x, 4)

    @staticmethod
    def F52022(x: ArrayLike) -> float:
        return CEC2022Functions._evaluate_cec2022(x, 5)

    @staticmethod
    def F62022(x: ArrayLike) -> float:
        return CEC2022Functions._evaluate_cec2022(x, 6)

    @staticmethod
    def F72022(x: ArrayLike) -> float:
        return CEC2022Functions._evaluate_cec2022(x, 7)

    @staticmethod
    def F82022(x: ArrayLike) -> float:
        return CEC2022Functions._evaluate_cec2022(x, 8)

    @staticmethod
    def F92022(x: ArrayLike) -> float:
        return CEC2022Functions._evaluate_cec2022(x, 9)

    @staticmethod
    def F102022(x: ArrayLike) -> float:
        return CEC2022Functions._evaluate_cec2022(x, 10)

    @staticmethod
    def F112022(x: ArrayLike) -> float:
        return CEC2022Functions._evaluate_cec2022(x, 11)

    @staticmethod
    def F122022(x: ArrayLike) -> float:
        return CEC2022Functions._evaluate_cec2022(x, 12)


class ClassicalFunctions:
    """经典基准函数集合"""

    @staticmethod
    def prod(it: ArrayLike) -> float:
        """计算迭代器中所有元素的乘积"""
        return float(np.prod(it))

    @staticmethod
    def Ufun(x: ArrayLike, a: float, k: float, m: float) -> ArrayLike:
        """部分基准函数中使用的惩罚函数"""
        x = np.asarray(x)
        return k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < (-a))

    @staticmethod
    def sphere(x: ArrayLike) -> float:
        """球函数(F1)"""
        return float(np.sum(np.square(x)))

    @staticmethod
    def F2(x: ArrayLike) -> float:
        """绝对值之和加乘积"""
        return float(np.sum(np.abs(x)) + ClassicalFunctions.prod(np.abs(x)))

    @staticmethod
    def F3(x: ArrayLike) -> float:
        """累积平方和"""
        x = np.asarray(x)
        return float(sum((np.sum(x[0:i + 1])) ** 2 for i in range(len(x))))

    @staticmethod
    def F4(x: ArrayLike) -> float:
        """最大绝对值"""
        return float(np.max(np.abs(x)))

    @staticmethod
    def rosenbrock(x: ArrayLike) -> float:
        """Rosenbrock函数(F5)"""
        x = np.asarray(x)
        return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))

    @staticmethod
    def F6(x: ArrayLike) -> float:
        """带偏移的平方和"""
        return float(np.sum(np.square(np.abs(x + 0.5))))

    @staticmethod
    def F7(x: ArrayLike) -> float:
        """带随机噪声的加权四次方和"""
        x = np.asarray(x)
        w = np.arange(1, len(x) + 1)
        return float(np.sum(w * x ** 4) + np.random.uniform(0, 1))

    @staticmethod
    def F8(x: ArrayLike) -> float:
        """改进的Schwefel函数"""
        return float(np.sum(-x * np.sin(np.sqrt(np.abs(x)))))

    @staticmethod
    def rastrigin(x: ArrayLike) -> float:
        """Rastrigin函数(F9)"""
        x = np.asarray(x)
        return float(RASTRIGIN_A * len(x) + np.sum(x ** 2 - RASTRIGIN_A * np.cos(2 * np.pi * x)))

    @staticmethod
    def ackley(x: ArrayLike) -> float:
        """Ackley函数(F10)"""
        x = np.asarray(x)
        d = len(x)
        sum1 = np.sum(x ** 2)
        sum2 = np.sum(np.cos(ACKLEY_C * x))
        return float(-ACKLEY_A * np.exp(-ACKLEY_B * np.sqrt(sum1 / d))
                    - np.exp(sum2 / d) + ACKLEY_A + np.e)

    @staticmethod
    def griewank(x: ArrayLike) -> float:
        """Griewank函数(F11)"""
        x = np.asarray(x)
        w = np.arange(1, len(x) + 1)
        return float(np.sum(x ** 2) / 4000
                    - ClassicalFunctions.prod(np.cos(x / np.sqrt(w))) + 1)


class SpecialFunctions:
    """特殊基准函数集合(F12-F23)"""

    @staticmethod
    def F12(x: ArrayLike) -> float:
        """惩罚函数1"""
        x = np.asarray(x)
        dim = len(x)
        y = 1 + (x + 1) / 4

        return float((np.pi / dim) * (
                10 * np.sin(np.pi * y[0]) ** 2 +
                np.sum((y[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * y[1:]) ** 2)) +
                (y[-1] - 1) ** 2
        ) + np.sum(ClassicalFunctions.Ufun(x, 10, 100, 4)))

    @staticmethod
    def F13(x: ArrayLike) -> float:
        """惩罚函数2"""
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        return float(0.1 * (
                np.sin(3 * np.pi * x[:, 0]) ** 2 +
                np.sum((x[:, :-1] - 1) ** 2 * (1 + np.sin(3 * np.pi * x[:, 1:]) ** 2), axis=1) +
                (x[:, -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * x[:, -1]) ** 2)
        ) + np.sum(ClassicalFunctions.Ufun(x, 5, 100, 4)))


def get_function_details(func_name: str) -> Union[List[Any], str]:
    """获取基准函数的详细信息

    参数:
        func_name: 函数名称

    返回:
        包含[名称, 下界, 上界, 维度]的列表，如果未找到则返回"nothing"
    """
    param = {
        "F1": ["F1", -100, 100, 30],
        "F2": ["F2", -10, 10, 30],
        "F3": ["F3", -100, 100, 30],
        "F4": ["F4", -100, 100, 30],
        "F5": ["F5", -30, 30, 30],
        "F6": ["F6", -100, 100, 30],
        "F7": ["F7", -1.28, 1.28, 30],
        "F8": ["F8", -500, 500, 30],
        "F9": ["F9", -5.12, 5.12, 30],
        "F10": ["F10", -32, 32, 30],
        "F11": ["F11", -600, 600, 30],
        "F12": ["F12", -50, 50, 30],
        "F13": ["F13", -50, 50, 30],
        "F14": ["F14", -65.536, 65.536, 2],
        "F15": ["F15", -5, 5, 4],
        "F16": ["F16", -5, 5, 2],
        "F17": ["F17", -5, 15, 2],
        "F18": ["F18", -2, 2, 2],
        "F19": ["F19", 0, 1, 3],
        "F20": ["F20", 0, 1, 6],
        "F21": ["F21", 0, 10, 4],
        "F22": ["F22", 0, 10, 4],
        "F23": ["F23", 0, 10, 4],
        "ackley": ["ackley", -32.768, 32.768, 30],  # Ackley函数
        "rosenbrock": ["rosenbrock", -5, 10, 30],  # Rosenbrock函数
        "rastrigin": ["rastrigin", -5.12, 5.12, 30],  # Rastrigin函数
        "griewank": ["griewank", -600, 600, 30],  # Griewank函数
        "F12022": ["F12022", -100, 100, 10],
        "F22022": ["F22022", -100, 100, 10],
        "F32022": ["F32022", -100, 100, 10],
        "F42022": ["F42022", -100, 100, 10],
        "F52022": ["F52022", -100, 100, 10],
        "F62022": ["F62022", -100, 100, 10],
        "F72022": ["F72022", -100, 100, 10],
        "F82022": ["F82022", -100, 100, 10],
        "F92022": ["F92022", -100, 100, 10],
        "F102022": ["F102022", -100, 100, 10],
        "F112022": ["F112022", -100, 100, 10],
        "F122022": ["F122022", -100, 100, 10]
    }
    return param.get(func_name, "nothing")


# 为保持向后兼容性
F1 = ClassicalFunctions.sphere
F2 = ClassicalFunctions.F2
F3 = ClassicalFunctions.F3
F4 = ClassicalFunctions.F4
F5 = ClassicalFunctions.rosenbrock
F6 = ClassicalFunctions.F6
F7 = ClassicalFunctions.F7
F8 = ClassicalFunctions.F8
F9 = ClassicalFunctions.rastrigin
F10 = ClassicalFunctions.ackley
F11 = ClassicalFunctions.griewank
F12 = SpecialFunctions.F12
F13 = SpecialFunctions.F13
F14 = lambda x: NotImplemented  # 需要时实现
F15 = lambda x: NotImplemented  # 需要时实现
F16 = lambda x: NotImplemented  # 需要时实现
F17 = lambda x: NotImplemented  # 需要时实现
F18 = lambda x: NotImplemented  # 需要时实现
F19 = lambda x: NotImplemented  # 需要时实现
F20 = lambda x: NotImplemented  # 需要时实现
F21 = lambda x: NotImplemented  # 需要时实现
F22 = lambda x: NotImplemented  # 需要时实现
F23 = lambda x: NotImplemented  # 需要时实现

# CEC 2022函数
F12022 = CEC2022Functions.F12022
F22022 = CEC2022Functions.F22022
F32022 = CEC2022Functions.F32022
F42022 = CEC2022Functions.F42022
F52022 = CEC2022Functions.F52022
F62022 = CEC2022Functions.F62022
F72022 = CEC2022Functions.F72022
F82022 = CEC2022Functions.F82022
F92022 = CEC2022Functions.F92022
F102022 = CEC2022Functions.F102022
F112022 = CEC2022Functions.F112022
F122022 = CEC2022Functions.F122022

# 额外的经典函数
ackley = ClassicalFunctions.ackley
rosenbrock = ClassicalFunctions.rosenbrock
rastrigin = ClassicalFunctions.rastrigin
griewank = ClassicalFunctions.griewank

# 重命名getFunctionDetails以符合PEP 8规范
getFunctionDetails = get_function_details