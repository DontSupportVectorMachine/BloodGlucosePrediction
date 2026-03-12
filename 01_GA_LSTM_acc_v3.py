# GPU加速torch（防止显存溢出）版本+早停实现+ 基因扰动——生理特征增强版
import random, re, glob, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from deap import base, creator, tools
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm
import time
import gc

start_time = time.time()  # ⏱ 记录开始时间
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"√ 当前设备: {device}")

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)
# ========== 早停机制类 (保持原样) ==========
class EarlyStopping:
    def __init__(self, patience=5, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


# ========== 模型定义 (保持原样) ==========
class LSTMModel(nn.Module):
    def __init__(self, in_feat, hidden=64, drop=0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(in_feat, hidden, batch_first=True)
        self.drop1 = nn.Dropout(drop)
        self.lstm2 = nn.LSTM(hidden, hidden, batch_first=True)
        self.drop2 = nn.Dropout(drop)
        self.fc1 = nn.Linear(hidden, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        o, _ = self.lstm1(x)
        o = self.drop1(o)
        o, _ = self.lstm2(o)
        o = self.drop2(o[:, -1, :])
        o = self.fc1(o)
        o = self.relu(o)
        return self.fc2(o)


class GRUModel(nn.Module):
    def __init__(self, in_feat, hidden=64, drop=0.3):
        super().__init__()
        self.gru1 = nn.GRU(in_feat, hidden, batch_first=True)
        self.drop1 = nn.Dropout(drop)
        self.gru2 = nn.GRU(hidden, hidden, batch_first=True)
        self.drop2 = nn.Dropout(drop)
        self.fc1 = nn.Linear(hidden, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        o, _ = self.gru1(x)
        o = self.drop1(o)
        o, _ = self.gru2(o)
        o = self.drop2(o[:, -1, :])
        o = self.fc1(o)
        o = self.relu(o)
        return self.fc2(o)


def build_frozen_lstm(in_feat):
    m = LSTMModel(in_feat)
    for name, p in m.named_parameters():
        if 'fc' not in name:
            p.requires_grad = False
    return m


# ========== 模型训练与预测 (保持原样) ==========
def train_and_predict_torch(model, X_tr, y_tr, X_te, y_te, epochs=10, bs=16):
    model = model.to(device)
    model.train()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    loss_fn = nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float())
    loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
    early_stopper = EarlyStopping(patience=3)
    for epoch in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb.unsqueeze(1))
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            X_te_tensor = torch.from_numpy(X_te).float().to(device)
            pred = model(X_te_tensor).cpu().numpy().flatten()
            val_loss = mean_squared_error(y_te, pred)
            early_stopper(val_loss)
            if early_stopper.early_stop:
                break
        model.train()

    model.eval()
    with torch.no_grad():
        X_te_tensor = torch.from_numpy(X_te).float().to(device)
        pred = model(X_te_tensor).cpu().numpy().flatten()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return pred


# ========== 数据预处理 (更新：适配 3 维特征) ==========
# 假设你的新数据保存在这个目录下
DATA_PATH = r"./processed_data_v2/Full"
file_paths = glob.glob(os.path.join(DATA_PATH, "*.csv"))  # 注意：之前脚本输出的是csv
dfs = []

print(f"🚀 正在加载生理特征数据集，共发现 {len(file_paths)} 个文件...")

for i, f in enumerate(file_paths):
    d = pd.read_csv(f)  # 修改为 read_csv
    d['Date'] = pd.to_datetime(d['Date'])
    d.set_index('Date', inplace=True)
    d['Patient_ID'] = i
    # 核心：确保包含新特征列
    required_cols = ['CGM (mg / dl)', 'GI_Impact_Factor', 'Insulin_Impact_Factor']
    for col in required_cols:
        if col not in d.columns: d[col] = 0.0

    dfs.append(d)

df_data = pd.concat(dfs).sort_index().fillna(0)
# df_data = df_data[~df_data.index.duplicated()]  !!!注意，这行代码可能会出问题，如果患者 A 和患者 B 都在同一个时间点（例如 2024-01-01 08:00）有记录，系统会随机删掉其中一个患者的数据。

# 保持原有的 Patient_ID 逻辑
# 注意：之前代码中对胰岛素的正则提取在预处理脚本里已经做过了，这里直接读取计算好的 Factor 即可

# ========== 初始化全局变量 (更新：features 变为 3 维) ==========
features = ["CGM (mg / dl)", "GI_Impact_Factor", "Insulin_Impact_Factor"]
window_size = 8
convergence_mse, convergence_r2, generation_params = [], [], []
best_mse, best_r2, best_ind = float('inf'), -float('inf'), None


# ========== 适应度评估函数 (更新：适配多维 MinMaxScaler) ==========
def eval_params(ind):
    global best_mse, best_r2, best_ind
    alpha, beta = ind
    print(f"！ 正在评估个体: alpha={alpha}, beta={beta}")

    # 增加一个 alpha 和 beta 之间的最小间距约束，防止模型区间重叠
    if alpha < 200 or alpha > 450 or beta < 600 or beta > 1200 or (beta - alpha) < 100:
        return (float('inf'),)

    ms, rs = [], []
    patient_ids = df_data["Patient_ID"].unique()

    # 随机抽样评估 (可选：如果全量太慢，可以随机抽 30% 患者加速 GA)
    # sample_patients = random.sample(list(patient_ids), k=min(30, len(patient_ids)))

    for pid in patient_ids:
        df = df_data[df_data.Patient_ID == pid][features]
        L = len(df)

        # 模型分配逻辑保持原样，in_feat 自动变为 3
        if L < alpha:
            model = build_frozen_lstm(len(features))
        elif L > beta:
            model = LSTMModel(len(features))
        else:
            model = GRUModel(len(features))

        # 归一化处理 (适配 3 列)
        scaler = MinMaxScaler()
        arr = scaler.fit_transform(df)

        X, y = [], []
        for i in range(len(arr) - window_size):
            X.append(arr[i:i + window_size])  # 取 8 个步长的 3 维特征
            y.append(arr[i + window_size, 0])  # 预测目标仅为第 0 列 (CGM)

        X, y = np.array(X), np.array(y)
        split = int(0.8 * len(X))
        if split < 5: continue  # 数据量过小则跳过

        y_pred = train_and_predict_torch(model, X[:split], y[:split], X[split:], y[split:])

        # 线性偏置修正 绝对不可以用！！！删掉！！！
        # bias = np.mean(y[split:] - y_pred)
        # y_pred += bias

        # 反归一化 (核心：需要构造 3 列 dummy 数组)
        def inverse_target(y_val):
            dummy = np.zeros((len(y_val), len(features)))
            dummy[:, 0] = y_val.flatten()
            return scaler.inverse_transform(dummy)[:, 0]

        y_true_orig = inverse_target(y[split:])
        y_pred_orig = inverse_target(y_pred)

        mse = mean_squared_error(y_true_orig, y_pred_orig)
        r2 = r2_score(y_true_orig, y_pred_orig)
        ms.append(mse)
        rs.append(r2)

    if not ms: return (float('inf'),)

    avg_m, avg_r = np.mean(ms), np.mean(rs)
    convergence_mse.append(avg_m)
    convergence_r2.append(avg_r)
    generation_params.append({"Alpha": alpha, "Beta": beta, "MSE": avg_m, "R2": avg_r})

    if avg_m < best_mse:
        best_mse, best_r2, best_ind = avg_m, avg_r, ind[:]

    print(f"√ 评估完: MSE={avg_m:.4f}, R2={avg_r:.4f} | 最优: alpha={best_ind[0]}, beta={best_ind[1]}")
    return (avg_m,)


# ========== 遗传算法设置 (保持原样) ==========
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# 1. 修正后的初始化与注册逻辑
def gen_indiv():
    a = random.randint(200, 450)
    b = random.randint(max(a + 50, 600), 1200) # 增加最小间距
    return creator.Individual([a, b])

def custom_mutate(ind):
    """ 修正：确保原地修改并返回元组 """
    if random.random() < 0.3:
        ind[0] = random.randint(200, 450)
    if random.random() < 0.3:
        ind[1] = random.randint(600, 1200)
    # 强制约束
    if ind[0] >= ind[1]:
        ind[1] = ind[0] + 50
    return ind,

toolbox = base.Toolbox()
toolbox.register("individual", gen_indiv)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", custom_mutate) # 修正变异注册
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", eval_params)

# ========== 精英交叉策略主循环 (保持原样) ==========
def eaEliteCross(pop, toolbox, cxpb, mutpb_init, ngen, elite=1):
    gene_var_ref = 3000
    min_mutpb, max_mutpb = 0.01, 0.5
    gene_var_history, mutpb_history = [], []
    mutpb = mutpb_init

    # 初始评估
    fits = list(map(toolbox.evaluate, pop))
    for ind, f in zip(pop, fits): ind.fitness.values = f

    for gen in range(1, ngen + 1):
        print(f"\n--- 第 {gen} 代进化 ---")
        off = list(map(toolbox.clone, toolbox.select(pop, len(pop) - elite)))
        elites = tools.selBest(pop, elite)
        pool = off + list(map(toolbox.clone, elites))

        # 多样性监控
        genes = np.array(pool)
        gene_var = np.var(genes[:, 0]) + np.var(genes[:, 1])
        gene_var_history.append(gene_var)

        # 自适应变异
        mutpb = max(min_mutpb, min(mutpb_init * (gene_var_ref / (gene_var + 1e-6)), max_mutpb))
        mutpb_history.append(mutpb)

        # 交叉变异
        for c1, c2 in zip(pool[::2], pool[1::2]):
            if random.random() < cxpb:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values
        for ind in pool:
            if random.random() < mutpb:
                toolbox.mutate(ind)
                del ind.fitness.values

        # 基因扰动逻辑
        from collections import Counter
        count = Counter(tuple(ind) for ind in pool)
        repeated = [t for t, c in count.items() if c >= 3]
        # 2. 修正 eaEliteCross 中的扰动范围
        for ind in pool:
            if tuple(ind) in repeated and random.random() < 0.5:
                # 修正：边界对齐 200-450 和 600-1200
                ind[0] = max(200, min(450, ind[0] + random.choice([-10, -5, 5, 10])))
                ind[1] = max(ind[0] + 50, min(1200, ind[1] + random.choice([-10, -5, 5, 10])))
                del ind.fitness.values
        # 重新评估失效个体
        invalid = [i for i in pool if not i.fitness.valid]
        for ind in invalid: ind.fitness.values = toolbox.evaluate(ind)

        pop[:] = tools.selBest(pool, len(pop))
        print(f"-> 第 {gen} 代最优 MSE: {tools.selBest(pop, 1)[0].fitness.values[0]:.4f}")

    # 保存为 Excel
    df_var_mut = pd.DataFrame({
        "Generation": range(1, ngen + 1),
        "Diversity": gene_var_history,
        "MutationRate": mutpb_history
    })
    df_var_mut.to_excel(r"./diversity_mutation.xlsx", index=False)
    print("√ 多样性与变异率趋势数据已保存为 Excel")

    return pop, gene_var_history, mutpb_history

# ========== 执行主流程 ==========
NGEN = 20      # 实验中发现收敛快，20代足够展现收敛曲线
POP_SIZE = 40  # 种群规模提升到40

population = toolbox.population(n=POP_SIZE)
final_pop, var_hist, mut_hist = eaEliteCross(
    population,
    toolbox,
    cxpb=0.7,        # 交叉率
    mutpb_init=0.3,  # 变异率
    ngen=NGEN,
    elite=1
)

# ========== 汇总与可视化 ==========
print(f"！最优参数组合： {best_ind}")
print(f"最优α = {best_ind[0]}, 最优β = {best_ind[1]}")
sorted_list = sorted(generation_params, key=lambda x: x['MSE'])
print("√ 最优10个参数组合（R2 与 MSE 排行）---精英交叉策略：")
for i, v in enumerate(sorted_list[:10], 1):
    print(f"排名 {i}: 最优α = {v['Alpha']}, 最优β = {v['Beta']}, R2 = {v['R2']:.4f}, MSE = {v['MSE']:.4f}")

print("~收敛曲线数据（MSE）:", convergence_mse)
print("~收敛曲线数据（R2）:", convergence_r2)
plt.figure(figsize=(10, 5))
plt.plot(convergence_mse, 'o-r')
plt.title("MSE 收敛")
plt.xlabel("Generation")
plt.ylabel("MSE")
plt.grid(True)
plt.tight_layout()
plt.show()
plt.figure(figsize=(10, 5))
plt.plot(convergence_r2, 's-b')
plt.title("R2 收敛")
plt.xlabel("Generation")
plt.ylabel("R2")
plt.grid(True)
plt.tight_layout()
plt.show()
pd.DataFrame(generation_params).to_excel(r"./ga_results_torch.xlsx", index=False)
print("√ 保存至 Excel 完成。")

# 记录时间
end_time = time.time()
elapsed = end_time - start_time
print(f"time： 本次运行耗时{time.strftime('%H:%M:%S', time.gmtime(elapsed))}")
