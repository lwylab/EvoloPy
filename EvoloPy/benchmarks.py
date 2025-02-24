# -*- coding: utf-8 -*-
import math
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
    def F1(x: ArrayLike) -> float:
        return float(np.sum(np.square(x)))

    @staticmethod
    def F2(x: ArrayLike) -> float:
        return float(np.sum(np.abs(x)) + ClassicalFunctions.prod(np.abs(x)))

    @staticmethod
    def F3(x: ArrayLike) -> float:
        x = np.asarray(x)
        return float(sum((np.sum(x[0:i + 1])) ** 2 for i in range(len(x))))

    @staticmethod
    def F4(x: ArrayLike) -> float:
        return float(np.max(np.abs(x)))

    @staticmethod
    def F5(x: ArrayLike) -> float:
        x = np.asarray(x)
        return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))

    @staticmethod
    def F6(x: ArrayLike) -> float:
        return float(np.sum(np.square(np.abs(x + 0.5))))

    @staticmethod
    def F7(x: ArrayLike) -> float:
        x = np.asarray(x)
        w = np.arange(1, len(x) + 1)
        return float(np.sum(w * x ** 4) + np.random.uniform(0, 1))

    @staticmethod
    def F8(x: ArrayLike) -> float:
        return float(np.sum(-x * np.sin(np.sqrt(np.abs(x)))))

    @staticmethod
    def F9(x: ArrayLike) -> float:
        x = np.asarray(x)
        return float(RASTRIGIN_A * len(x) + np.sum(x ** 2 - RASTRIGIN_A * np.cos(2 * np.pi * x)))

    @staticmethod
    def F10(x: ArrayLike) -> float:
        x = np.asarray(x)
        d = len(x)
        sum1 = np.sum(x ** 2)
        sum2 = np.sum(np.cos(ACKLEY_C * x))
        return float(-ACKLEY_A * np.exp(-ACKLEY_B * np.sqrt(sum1 / d))
                     - np.exp(sum2 / d) + ACKLEY_A + np.e)

    @staticmethod
    def F11(x: ArrayLike) -> float:
        x = np.asarray(x)
        w = np.arange(1, len(x) + 1)
        return float(np.sum(x ** 2) / 4000
                     - ClassicalFunctions.prod(np.cos(x / np.sqrt(w))) + 1)


# define the function blocks
def prod(it):
    p = 1
    for n in it:
        p *= n
    return p


def Ufun(x, a, k, m):
    y = k * ((x - a) ** m) * (x > a) + k * ((-x - a) ** m) * (x < (-a))
    return y


def F1(x):
    s = np.sum(x ** 2)
    return s


def F2(x):
    o = sum(abs(x)) + prod(abs(x))
    return o


def F3(x):
    dim = len(x) + 1
    o = 0
    for i in range(1, dim):
        o = o + (np.sum(x[0:i])) ** 2
    return o


def F4(x):
    o = max(abs(x))
    return o


def F5(x):
    dim = len(x)
    o = np.sum(
        100 * (x[1:dim] - (x[0 : dim - 1] ** 2)) ** 2 + (x[0 : dim - 1] - 1) ** 2
    )
    return o


def F6(x):
    o = np.sum(abs((x + 0.5)) ** 2)
    return o


def F7(x):
    dim = len(x)
    w = np.arange(1, dim + 1)  # create an array from 1 to dim
    o = np.sum(w * (x ** 4)) + np.random.uniform(0, 1)
    return o


def F8(x):
    o = sum(-x * (np.sin(np.sqrt(abs(x)))))
    return o


def F9(x):
    dim = len(x)
    o = np.sum(x ** 2 - 10 * np.cos(2 * math.pi * x)) + 10 * dim
    return o


def F10(x):
    dim = len(x)
    o = (
        -20 * np.exp(-0.2 * np.sqrt(np.sum(x ** 2) / dim))
        - np.exp(np.sum(np.cos(2 * math.pi * x)) / dim)
        + 20
        + np.exp(1)
    )
    return o


def F11(x):
    dim = len(x)
    w = [i for i in range(len(x))]
    w = [i + 1 for i in w]
    o = np.sum(x ** 2) / 4000 - prod(np.cos(x / np.sqrt(w))) + 1
    return o


def F12(x):
    dim = len(x)
    o = (math.pi / dim) * (
        10 * ((np.sin(math.pi * (1 + (x[0] + 1) / 4))) ** 2)
        + np.sum(
            (((x[: dim - 1] + 1) / 4) ** 2)
            * (1 + 10 * ((np.sin(math.pi * (1 + (x[1 :] + 1) / 4)))) ** 2)
        )
        + ((x[dim - 1] + 1) / 4) ** 2
    ) + np.sum(Ufun(x, 10, 100, 4))
    return o


def F13(x):
    if x.ndim==1:
        x = x.reshape(1,-1)

    o = 0.1 * (
        (np.sin(3 * np.pi * x[:,0])) ** 2
        + np.sum(
            (x[:,:-1] - 1) ** 2
            * (1 + (np.sin(3 * np.pi * x[:,1:])) ** 2), axis=1
        )
        + ((x[:,-1] - 1) ** 2) * (1 + (np.sin(2 * np.pi * x[:,-1])) ** 2)
    ) + np.sum(Ufun(x, 5, 100, 4))
    return o


def F14(x):
    aS = [
        [
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
            -32, -16, 0, 16, 32,
        ],
        [
            -32, -32, -32, -32, -32,
            -16, -16, -16, -16, -16,
            0, 0, 0, 0, 0,
            16, 16, 16, 16, 16,
            32, 32, 32, 32, 32,
        ],
    ]
    aS = np.asarray(aS)
    bS = np.zeros(25)
    v = np.array(x)
    for i in range(0, 25):
        H = v - aS[:, i]
        bS[i] = np.sum((np.power(H, 6)))
    w = np.arange(1, 26)
    o = ((1.0 / 500) + np.sum(1.0 / (w + bS))) ** (-1)
    return o


def F15(L):
    aK = [
        0.1957, 0.1947, 0.1735, 0.16, 0.0844, 0.0627,
        0.0456, 0.0342, 0.0323, 0.0235, 0.0246,
    ]
    bK = [0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16]
    aK = np.asarray(aK)
    bK = np.asarray(bK)
    bK = 1 / bK
    fit = np.sum(
        (aK - ((L[0] * (bK ** 2 + L[1] * bK)) / (bK ** 2 + L[2] * bK + L[3]))) ** 2
    )
    return fit


def F16(L):
    o = (
        4 * (L[0] ** 2)
        - 2.1 * (L[0] ** 4)
        + (L[0] ** 6) / 3
        + L[0] * L[1]
        - 4 * (L[1] ** 2)
        + 4 * (L[1] ** 4)
    )
    return o


def F17(L):
    o = (
        (L[1] - (L[0] ** 2) * 5.1 / (4 * (np.pi ** 2)) + 5 / np.pi * L[0] - 6)
        ** 2
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(L[0])
        + 10
    )
    return o


def F18(L):
    o = (
        1
        + (L[0] + L[1] + 1) ** 2
        * (
            19
            - 14 * L[0]
            + 3 * (L[0] ** 2)
            - 14 * L[1]
            + 6 * L[0] * L[1]
            + 3 * L[1] ** 2
        )
    ) * (
        30
        + (2 * L[0] - 3 * L[1]) ** 2
        * (
            18
            - 32 * L[0]
            + 12 * (L[0] ** 2)
            + 48 * L[1]
            - 36 * L[0] * L[1]
            + 27 * (L[1] ** 2)
        )
    )
    return o


# map the inumpyuts to the function blocks
def F19(L):
    aH = [[3, 10, 30], [0.1, 10, 35], [3, 10, 30], [0.1, 10, 35]]
    aH = np.asarray(aH)
    cH = [1, 1.2, 3, 3.2]
    cH = np.asarray(cH)
    pH = [
        [0.3689, 0.117, 0.2673],
        [0.4699, 0.4387, 0.747],
        [0.1091, 0.8732, 0.5547],
        [0.03815, 0.5743, 0.8828],
    ]
    pH = np.asarray(pH)
    o = 0
    for i in range(0, 4):
        o = o - cH[i] * np.exp(-(np.sum(aH[i, :] * ((L - pH[i, :]) ** 2))))
    return o


def F20(L):
    aH = [
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14],
    ]
    aH = np.asarray(aH)
    cH = [1, 1.2, 3, 3.2]
    cH = np.asarray(cH)
    pH = [
        [0.1312, 0.1696, 0.5569, 0.0124, 0.8283, 0.5886],
        [0.2329, 0.4135, 0.8307, 0.3736, 0.1004, 0.9991],
        [0.2348, 0.1415, 0.3522, 0.2883, 0.3047, 0.6650],
        [0.4047, 0.8828, 0.8732, 0.5743, 0.1091, 0.0381],
    ]
    pH = np.asarray(pH)
    o = 0
    for i in range(0, 4):
        o = o - cH[i] * np.exp(-(np.sum(aH[i, :] * ((L - pH[i, :]) ** 2))))
    return o


def F21(L):
    aSH = [
        [4, 4, 4, 4],
        [1, 1, 1, 1],
        [8, 8, 8, 8],
        [6, 6, 6, 6],
        [3, 7, 3, 7],
        [2, 9, 2, 9],
        [5, 5, 3, 3],
        [8, 1, 8, 1],
        [6, 2, 6, 2],
        [7, 3.6, 7, 3.6],
    ]
    cSH = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    aSH = np.asarray(aSH)
    cSH = np.asarray(cSH)
    fit = 0
    for i in range(5):
        v = np.array(L - aSH[i, :])
        fit = fit - ((v) * (v.T) + cSH[i]) ** (-1)
    o = fit.item(0)
    return o


def F22(L):
    aSH = [
        [4, 4, 4, 4],
        [1, 1, 1, 1],
        [8, 8, 8, 8],
        [6, 6, 6, 6],
        [3, 7, 3, 7],
        [2, 9, 2, 9],
        [5, 5, 3, 3],
        [8, 1, 8, 1],
        [6, 2, 6, 2],
        [7, 3.6, 7, 3.6],
    ]
    cSH = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    aSH = np.asarray(aSH)
    cSH = np.asarray(cSH)
    fit = 0
    for i in range(7):
        v = np.array(L - aSH[i, :])
        fit = fit - ((v) * (v.T) + cSH[i]) ** (-1)
    o = fit.item(0)
    return o


def F23(L):
    aSH = [
        [4, 4, 4, 4],
        [1, 1, 1, 1],
        [8, 8, 8, 8],
        [6, 6, 6, 6],
        [3, 7, 3, 7],
        [2, 9, 2, 9],
        [5, 5, 3, 3],
        [8, 1, 8, 1],
        [6, 2, 6, 2],
        [7, 3.6, 7, 3.6],
    ]
    cSH = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5]
    aSH = np.asarray(aSH)
    cSH = np.asarray(cSH)
    fit = 0
    for i in range(10):
        v = np.array(L - aSH[i, :])
        fit = fit - ((v) * (v.T) + cSH[i]) ** (-1)
    o = fit.item(0)
    return o

# Ackley function (commonly used in optimization)
def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + a + np.e

# Rosenbrock function (tests convergence)
def rosenbrock(x):
    return sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

# Rastrigin function (tests local minima avoidance)
def rastrigin(x):
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))

# Griewank function (tests exploration)
def griewank(x):
    part1 = np.sum(x**2) / 4000
    part2 = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return part1 - part2 + 1

def getFunctionDetails(func_name: str) -> Union[List[Any], str]:
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
