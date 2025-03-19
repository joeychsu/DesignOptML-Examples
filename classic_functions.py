import numpy as np

# Ackley 函數：一個多峰函數，具有多個局部最小值，常用於測試最佳化演算法在複雜搜索空間中的表現。
def ackley(x):
    x = np.array(x)  # 將列表轉換為 NumPy 陣列
    a = 20
    b = 0.2
    c = 2 * np.pi
    n = len(x)  # 參數向量的維度
    sum1 = -a * np.exp(-b * np.sqrt(np.sum(np.square(x)) / n))
    sum2 = -np.exp(np.sum(np.cos(c * x) / n))
    return sum1 + sum2 + a + np.exp(1)

# Rastrigin 函數：一個多峰函數，具有大量局部最小值，常用於測試最佳化演算法的全局搜索能力。
def rastrigin(x):
    x = np.array(x)  # 將列表轉換為 NumPy 陣列
    A = 10
    n = len(x)  # 參數向量的維度
    sum_term = np.sum(x**2 - A * np.cos(2 * np.pi * x))
    return A * n + sum_term

# Rosenbrock 函數：一個非凸函數，具有狹長的谷底，常用於測試最佳化演算法在平坦區域的表現。
def rosenbrock(x):
    x = np.array(x)  # 將列表轉換為 NumPy 陣列
    n = len(x)  # 參數向量的維度
    sum_term = 0
    for i in range(n - 1):
        sum_term += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return sum_term

# Sphere 函數：一個簡單的凸函數，常用於測試最佳化演算法的基本收斂性能。
def sphere(x):
    x = np.array(x)  # 將列表轉換為 NumPy 陣列
    return np.sum(x**2)

def get_func(func_name):
    """
    根據 func_name 返回對應的測試函數。
    
    參數:
    func_name (str): 測試函數的名稱，支持 "ackley", "rastrigin", "rosenbrock", "sphere"。
    
    返回:
    function: 對應的測試函數。
    
    引發:
    ValueError: 如果 func_name 不在支援的列表中。
    """
    if func_name == "ackley":
        return ackley
    elif func_name == "rastrigin":
        return rastrigin
    elif func_name == "rosenbrock":
        return rosenbrock
    elif func_name == "sphere":
        return sphere
    else:
        raise ValueError(f"未知的函數名稱: {func_name}")