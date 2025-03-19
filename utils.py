from skopt.space import Real, Integer
import numpy as np
import argparse

class DefaultHelpFormatter(argparse.HelpFormatter):
    def _format_action(self, action):
        # 獲取原始的 help 訊息
        help_text = action.help if action.help is not None else ""
        
        # 如果有預設值，則在 help 後面添加 "(default: xxx)"
        if action.default is not None:
            help_text += f" (default: {action.default})"
        
        # 將修改後的 help 訊息設置回 action
        action.help = help_text
        
        # 調用父類方法生成參數描述
        parts = super()._format_action(action)
        
        return parts
        
class DiscreteReal(Real):
    """
    自訂的 DiscreteReal 類別，繼承自 skopt.space.Real，用於定義帶有步長限制的連續參數。
    
    參數:
        low (float): 參數的下界。
        high (float): 參數的上界。
        step (float): 參數的步長（離散化間隔）。
        name (str, optional): 參數的名稱，預設為 None。
    """
    def __init__(self, low, high, step, name=None):
        super().__init__(low, high, name=name)
        self.step = step
 
    def rvs(self, n_samples=1, random_state=None):
        """
        生成隨機樣本，並將其離散化到指定的步長倍數。
        
        參數:
            n_samples (int): 要生成的樣本數量，預設為 1。
            random_state (int or RandomState, optional): 隨機種子，用於控制隨機性，預設為 None。
        
        返回:
            np.ndarray: 離散化後的隨機樣本陣列。
        """
        # 將 list 轉換為 NumPy 陣列
        values = np.array(super().rvs(n_samples, random_state))
        # 進行離散化處理，將值調整到步長的倍數
        discrete_values = np.round(values / self.step) * self.step
        return discrete_values

def discretize_params(params, space):
    """
    根據參數空間的配置，對參數列表進行離散化處理。
    
    參數:
        params (list): 要離散化的參數列表。
        space (list): 參數空間的定義，包含每個參數的類型和屬性。
    
    返回:
        list: 離散化後的參數列表。
    """
    discretized = []
    for p, dim in zip(params, space):
        if isinstance(dim, DiscreteReal):
            step = dim.step
            p = round(p / step) * step  # 離散化到步長倍數
        elif isinstance(dim, Integer):
            p = round(p)  # 整數化處理
        else:
            p = p  # 其他類型保持不變
        discretized.append(p)
    return discretized