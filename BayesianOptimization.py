import numpy as np
from skopt import Optimizer
from skopt.space import Real, Integer
import time
from classic_functions import get_func
from utils import DiscreteReal, discretize_params, DefaultHelpFormatter
import argparse

def main():
    # 設定命令行參數解析器
    parser = argparse.ArgumentParser(description="貝式優化程式", formatter_class=DefaultHelpFormatter)
    parser.add_argument("--space_dim", type=int, default=5, help="參數空間的維度")
    parser.add_argument("--seeds", type=int, nargs='+', default=[42, 123, 456, 789, 101112], help="隨機種子的列表")
    parser.add_argument("--n_samples", type=int, default=10, help="初始參數的數量")
    parser.add_argument("--optimizer_type", type=str, default="GP", choices=["GP", "RF"], help="優化器類型：'GP' (高斯過程) 或 'RF' (隨機森林)")
    parser.add_argument("--n_iterations", type=int, default=100, help="迭代次數")
    parser.add_argument("--func_name", type=str, default="ackley", help="測試函數的名稱")

    # 解析命令行參數
    args = parser.parse_args()

    # 列出此次執行的設定值
    print("執行設定：")
    print(f"  參數空間維度: {args.space_dim}")
    print(f"  隨機種子列表: {args.seeds}")
    print(f"  初始參數數量: {args.n_samples}")
    print(f"  優化器類型: {args.optimizer_type}")
    print(f"  迭代次數: {args.n_iterations}")
    print(f"  測試函數: {args.func_name}")

    # 獲取測試函數
    func = get_func(args.func_name)

    # 動態生成參數空間
    space = []
    for i in range(args.space_dim):
        if i % 2 == 0:
            space.append(DiscreteReal(-5.0, 5.0, step=0.01, name=f'param{i+1}'))
        else:
            space.append(Integer(-5, 5, name=f'param{i+1}'))

    # 用於儲存每次運行的最佳值和耗時
    best_values = []
    total_times = []
    final_best_iterations = []  # 用於儲存每次運行中最終最佳值出現的迭代次數

    # 運行多次不同 seed 的貝式優化
    for run, seed in enumerate(args.seeds):
        np.random.seed(seed)  # 設定隨機種子
        print(f"\n=== 第 {run+1} 次運行 (seed={seed}) ===")

        # 準備初始樣本
        X_init = []
        y_init = []
        for _ in range(args.n_samples):
            sample = [np.random.uniform(-5.0, 5.0) if isinstance(space[i], DiscreteReal) else np.random.randint(-5, 6) 
                      for i in range(len(space))]
            X_init.append(sample)
            y_init.append(func(sample))

        # 初始化貝式優化器
        if args.optimizer_type == "RF":
            optimizer = Optimizer(
                space,
                base_estimator="RF",          # 使用隨機森林作為代理模型
                acq_func="EI",                # 期望改進作為採集函數
                acq_optimizer="sampling",     # 隨機採樣優化採集函數
                random_state=seed
            )
        else:  # GP
            optimizer = Optimizer(
                space,
                base_estimator="GP",          # 使用高斯過程
                acq_func="EI",                # 期望改進
                acq_optimizer="lbfgs",        # L-BFGS 優化採集函數
                random_state=seed
            )

        # 回饋初始資料
        for x, y in zip(X_init, y_init):
            optimizer.tell(x, y)

        # 開始計時
        total_start_time = time.time()
        current_best = float('inf')  # 假設是最小化問題
        last_best_iteration = 0  # 記錄此次運行中最終最佳值出現的迭代次數

        # 執行迭代
        for i in range(args.n_iterations):
            next_params = optimizer.ask()
            next_params_discrete = discretize_params(next_params, space)
            result = func(next_params_discrete)
            optimizer.tell(next_params_discrete, result)

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
        best_params = optimizer.Xi[np.argmin(optimizer.yi)]
        best_value = min(optimizer.yi)
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

if __name__ == "__main__":
    main()