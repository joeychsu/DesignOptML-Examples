import numpy as np
import argparse
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize
import time
from utils import DefaultHelpFormatter
from classic_functions import get_func

def get_surrogate_model(model_type, seed):
    """根據模型類型返回相應的代理模型"""
    if model_type == "mlp":
        from sklearn.neural_network import MLPRegressor
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        return Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(hidden_layer_sizes=(128, 128, 64), max_iter=3000, random_state=seed))
        ])
    elif model_type == "rf":
        from sklearn.ensemble import RandomForestRegressor
        return Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestRegressor(n_estimators=100, random_state=seed))
        ])
    elif model_type == "svr":
        from sklearn.svm import SVR
        return Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(kernel='rbf', C=1.0, epsilon=0.1))
        ])
    elif model_type == "gp":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        return Pipeline([
            ('scaler', StandardScaler()),
            ('gp', GaussianProcessRegressor(kernel=RBF(), n_restarts_optimizer=10, random_state=seed))
        ])
    else:
        raise ValueError(f"未知的模型類型: {model_type}")

# 離散化參數（根據需求調整參數範圍）
def discretize_params(params, bounds, step=0.01):
    return [np.round(p / step) * step if i % 2 == 0 else int(p) for i, (p, (low, high)) in enumerate(zip(params, bounds))]

def main():
    # 設定命令行參數解析器
    parser = argparse.ArgumentParser(description="純粹代理模型優化 (SBO)", formatter_class=DefaultHelpFormatter)
    parser.add_argument("--space_dim", type=int, default=5, help="參數空間的維度")
    parser.add_argument("--seeds", type=int, nargs='+', default=[42, 123, 456, 789, 101112], help="隨機種子的列表")
    parser.add_argument("--n_samples", type=int, default=10, help="初始參數的數量")
    parser.add_argument("--n_iterations", type=int, default=100, help="迭代次數")
    parser.add_argument("--model_type", type=str, default="mlp", choices=["mlp", "rf", "svr", "gp"], help="代理模型的類型")
    parser.add_argument("--func_name", type=str, default="ackley", help="測試函數的名稱")
    args = parser.parse_args()

    # 顯示執行設定
    print("------------------------------------------------------")
    print("執行設定：")
    print(f"  參數空間維度: {args.space_dim}")
    print(f"  隨機種子列表: {args.seeds}")
    print(f"  初始參數數量: {args.n_samples}")
    print(f"  迭代次數: {args.n_iterations}")
    print(f"  測試函數: {args.func_name}")
    print(f"  代理模型類型: {args.model_type}")

    # 獲取測試函數
    func = get_func(args.func_name)

    # 定義參數空間的邊界
    bounds = [(-5.0, 5.0) if i % 2 == 0 else (-5, 5) for i in range(args.space_dim)]

    # 用於儲存每次運行的最佳值和耗時
    best_values = []
    total_times = []
    final_best_iterations = []  # 用於儲存每次運行中最終最佳值出現的迭代次數

    # 多次運行，使用不同隨機種子
    for run, seed in enumerate(args.seeds):
        np.random.seed(seed)
        print(f"\n=== 第 {run+1} 次運行 (seed={seed}) ===")

        # 生成初始樣本
        X_init = np.random.uniform(-5, 5, (args.n_samples, args.space_dim))
        y_init = np.array([func(x) for x in X_init])

        # 初始化代理模型
        surrogate_model = get_surrogate_model(args.model_type, seed)
        surrogate_model.fit(X_init, y_init)

        # 記錄當前最佳值和計時
        current_best = np.min(y_init)
        total_start_time = time.time()

        # 定義代理模型的目標函數
        def surrogate_objective(x):
            x = np.atleast_2d(x)
            return surrogate_model.predict(x)[0]

        # 迭代優化
        for i in range(args.n_iterations):
            # 在代理模型上尋找預測的最佳點
            result = minimize(surrogate_objective, x0=np.random.uniform(-5, 5, args.space_dim), 
                            bounds=bounds, method='L-BFGS-B')
            next_params = result.x
            next_params_discrete = discretize_params(next_params, bounds)
            result = func(next_params_discrete)

            # 更新數據集
            X_init = np.vstack([X_init, next_params_discrete])
            y_init = np.append(y_init, result)

            # 重新訓練代理模型
            surrogate_model.fit(X_init, y_init)

            # 檢查是否有新的最佳值
            if result < current_best:
                current_best = result
                last_best_iteration = i + 1  # 更新最終最佳值的迭代次數
                current_time = time.time() - total_start_time
                print(f"新的最佳值出現！迭代次數: {i+1}, 最佳值: {result:.4f}, 目前耗時: {current_time:.2f} 秒")

            # 每 50 次迭代顯示進度
            if (i + 1) % 50 == 0:
                current_time = time.time() - total_start_time
                print(f"迭代次數: {i+1}, 當前最佳值: {current_best:.4f}, 目前耗時: {current_time:.2f} 秒")

        # 最終結果
        best_idx = np.argmin(y_init)
        best_params = X_init[best_idx]
        best_value = y_init[best_idx]
        final_time = time.time() - total_start_time
        print(f"最佳參數: {best_params}, 最佳結果: {best_value:.4f}, 總耗時: {final_time:.2f} 秒")

        # 儲存結果
        best_values.append(best_value)
        total_times.append(final_time)
        if last_best_iteration > 0:
            final_best_iterations.append(last_best_iteration)  # 記錄此次運行的最終最佳迭代次數
        else:
            print("此次運行中未出現新的最佳值")

    # 計算並顯示平均結果
    avg_best_value = np.mean(best_values)
    avg_total_time = np.mean(total_times)
    print(f"\n=== {len(args.seeds)} 次運行的平均結果 ===")
    print(f"平均最佳值: {avg_best_value:.4f}")
    print(f"平均總耗時: {avg_total_time:.2f} 秒")

    # 計算並顯示最終最佳值出現的平均迭代次數
    if final_best_iterations:
        avg_final_best_iteration = np.mean(final_best_iterations)
        print(f"最終最佳值出現的平均迭代次數: {avg_final_best_iteration:.2f}")
    else:
        print("在所有運行中未出現新的最佳值")

    print("------------------------------------------------------")

if __name__ == "__main__":
    main()