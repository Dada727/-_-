import random
import math
import os
import time
import statistics
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# ================== 四种混沌映射 ==================
def logistic(x, mu=3.99):
    return mu * x * (1 - x)

def tent(x, mu=1.999):
    return 1 - 2 * abs(x - 0.5)

def sine(x, mu=0.87):
    return mu * math.sin(math.pi * x)

def chebyshev(x, k=4):
    return math.cos(k * math.acos(x))

# ================== 生成置乱表 ==================
def generate_permutation(N, map_func, x0, params, discard=1000, eps=1e-15):
    x = x0
    for _ in range(discard):
        x = map_func(x, **params)
    seq = []
    for _ in range(N):
        x = map_func(x, **params)
        seq.append(x + eps * random.random())
    indexed = [(value, idx) for idx, value in enumerate(seq)]
    indexed.sort(key=lambda t: t[0])
    perm = [0] * N
    for rank, (_, original_idx) in enumerate(indexed, start=1):
        perm[original_idx] = rank
    return perm

# ================== 循环分析 ==================
def cycle_decomposition(perm):
    n = len(perm)
    visited = [False] * n
    cycles = []
    for i in range(n):
        if not visited[i]:
            length = 0
            j = i
            while not visited[j]:
                visited[j] = True
                j = perm[j] - 1
                length += 1
            cycles.append(length)
    return cycles

def order_of_permutation(cycles):
    lcm = 1
    for L in cycles:
        lcm = lcm * L // math.gcd(lcm, L)
    return lcm

# ================== 计算一组随机种子的阶列表 ==================
def orders_for_N(N, map_func, params, num_seeds=30, discard=1000):
    orders = []
    for _ in range(num_seeds):
        if map_func == chebyshev:
            x0 = random.uniform(-1, 1)
        else:
            x0 = random.random()
        perm = generate_permutation(N, map_func, x0, params, discard)
        cycles = cycle_decomposition(perm)
        orders.append(order_of_permutation(cycles))
    return orders

# ================== 统计量计算 ==================
def compute_stats(orders):
    mean_arith = sum(orders) / len(orders)
    log_orders = [math.log(o) for o in orders]
    mean_geo = math.exp(statistics.mean(log_orders))
    median = statistics.median(orders)
    std_log = statistics.stdev(log_orders) if len(orders) > 1 else 0.0
    return mean_arith, mean_geo, median, std_log

# ================== 主程序（带计时） ==================
def main():
    N_list = [100, 200, 500, 1000, 2000]
    num_seeds = 30
    discard = 1000

    maps = {
        "Logistic": (logistic, {"mu": 3.99}),
        "Tent":     (tent,     {"mu": 1.999}),
        "Sine":     (sine,     {"mu": 0.87}),
        "Chebyshev":(chebyshev,{"k": 4})
    }

    results = {name: [] for name in maps}
    all_orders = {name: [] for name in maps}
    timing = {name: [] for name in maps}  # 存储每个N的耗时（秒）

    for name, (func, params) in maps.items():
        print(f"\n===== 映射: {name} =====")
        total_start = time.perf_counter()
        for N in N_list:
            start = time.perf_counter()
            orders = orders_for_N(N, func, params, num_seeds, discard)
            elapsed = time.perf_counter() - start
            timing[name].append(elapsed)
            mean_arith, mean_geo, median, std_log = compute_stats(orders)
            results[name].append((mean_arith, mean_geo, median, std_log))
            all_orders[name].append(orders)
            print(f"N={N:4d} : 算术平均={mean_arith:.2e}, 几何平均={mean_geo:.2e}, "
                  f"中位数={median:.2e}, 对数标准差={std_log:.3f}, 耗时={elapsed:.3f}秒")
        total_elapsed = time.perf_counter() - total_start
        print(f"  映射 {name} 总耗时: {total_elapsed:.2f} 秒")

    # ================== 汇总打印整体性能 ==================
    print("\n========== 性能汇总 ==========")
    for name in maps:
        total_time = sum(timing[name])
        print(f"{name:10s} : 各N耗时 {['{:.3f}'.format(t) for t in timing[name]]} , 总和={total_time:.2f}秒")
    print(f"整体运行时间: {sum(sum(t) for t in timing.values()):.2f} 秒")

    # ================== 绘图（保持不变） ==================
    save_dir = "./cryp_tu"
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 7))
    colors = ['red', 'blue', 'green', 'orange']
    markers = ['o', 's', '^', 'D']
    for (name, stats_list), color, marker in zip(results.items(), colors, markers):
        N_arr = np.array(N_list)
        geo_means = [stats[1] for stats in stats_list]
        std_logs = [stats[3] for stats in stats_list]
        upper = [geo * math.exp(std) for geo, std in zip(geo_means, std_logs)]
        lower = [geo / math.exp(std) for geo, std in zip(geo_means, std_logs)]
        plt.plot(N_arr, geo_means, marker=marker, linestyle='-', color=color,
                 linewidth=2, markersize=6, label=f"{name} (geometric mean)")
        plt.fill_between(N_arr, lower, upper, color=color, alpha=0.2)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Permutation size N", fontsize=12)
    plt.ylabel("Order of permutation", fontsize=12)
    plt.title("Geometric mean of order with ±1σ error band (log-space)")
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(save_dir, "avg_order_with_error.png")
    plt.savefig(save_path, dpi=150)
    print(f"\n图片已保存至: {save_path}")
    plt.show()

if __name__ == "__main__":
    main()