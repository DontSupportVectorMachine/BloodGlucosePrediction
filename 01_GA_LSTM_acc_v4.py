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

# ========== 固定所有随机种子，保证实验完全可复现 ==========
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

start_time = time.time()  # 记录开始时间
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"√ 当前设备: {device}")


# ========== 早停机制类 ==========
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


# ========== 模型定义 ==========
# ========== 模型定义 ==========
class LSTMModel(nn.Module):
    def __init__(self, in_feat, hidden=64, drop=0.3):
        super().__init__()
        self.lstm1 = nn.LSTM(in_feat, hidden, batch_first=True)
        self.lstm2 = nn.LSTM(hidden, hidden, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Linear(hidden, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        o, _ = self.lstm1(x)
        o, _ = self.lstm2(o)
        o = self.drop(o[:, -1, :])
        o = self.fc1(o)
        o = self.relu(o)

        delta = self.fc2(o)  # 网络现在只需要专心学习 "波动变化量 (Delta)"

        # x[:, -1, 0] 是当前窗口最后一个时间点的真实已知血糖值 (归一化状态)
        last_known_cgm = x[:, -1, 0].unsqueeze(1)

        # 最终预测 = 当前已知血糖 + 模型预测的波动量
        return last_known_cgm + delta


class GRUModel(nn.Module):
    def __init__(self, in_feat, hidden=64, drop=0.3):
        super().__init__()
        self.gru = nn.GRU(in_feat, hidden, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Linear(hidden, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        o, _ = self.gru(x)
        o = self.drop(o[:, -1, :])
        o = self.fc1(o)
        o = self.relu(o)

        delta = self.fc2(o)

        last_known_cgm = x[:, -1, 0].unsqueeze(1)
        return last_known_cgm + delta


def build_frozen_lstm(in_feat):
    m = LSTMModel(in_feat)
    for name, p in m.named_parameters():
        if 'fc' not in name:
            p.requires_grad = False
    return m

# ========== 模型训练与预测 (修正显存管理) ==========
def train_and_predict_torch(model, X_tr, y_tr, X_val, y_val, X_te, epochs=10, bs=16):
    model = model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    loss_fn = nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float())
    loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
    early_stopper = EarlyStopping(patience=5)

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb.unsqueeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            X_val_tensor = torch.from_numpy(X_val).float().to(device)
            pred_v = model(X_val_tensor)
            val_loss = loss_fn(pred_v, torch.from_numpy(y_val).float().to(device).unsqueeze(1)).item()
            early_stopper(val_loss)
            if early_stopper.early_stop:
                break

    model.eval()
    with torch.no_grad():
        X_val_tensor = torch.from_numpy(X_val).float().to(device)
        pred_val = model(X_val_tensor).cpu().numpy().flatten()
        X_te_tensor = torch.from_numpy(X_te).float().to(device)
        pred_te = model(X_te_tensor).cpu().numpy().flatten()

    # 强制释放资源，防止 GA 迭代中显存爆炸
    model.to('cpu')
    del model, optimizer, loader, dataset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return pred_val, pred_te


# ========== 数据预处理 (修正数据交织) ==========
DATA_PATH = r"./processed_data_v2/Full"
file_paths = glob.glob(os.path.join(DATA_PATH, "*.csv"))
patient_data_dict = {}  # 改用字典存储，防止跨样本污染
features = ["CGM (mg / dl)", "GI_Impact_Factor", "Insulin_Impact_Factor"]

print(f" 正在加载生理特征数据集，共发现 {len(file_paths)} 个文件...")

all_train_dfs = []  # 用于拟合全局统一的 Scaler

for i, f in enumerate(file_paths):
    d = pd.read_csv(f, parse_dates=['Date'])
    d = d.sort_values('Date').set_index('Date')

    if 'CGM (mg / dl)' not in d.columns: continue

    # 患者内部独立填充，严禁跨患者填充
    d = d.ffill().fillna(0)
    patient_data_dict[i] = d[features]

    # 提取每个患者的前 70% 作为全局归一化基准的拟合源（严谨做法）
    train_cut = int(len(d) * 0.7)
    all_train_dfs.append(d[features].iloc[:train_cut])

# 【核心修正】全局统一 Scaler，消除基准漂移
global_scaler = MinMaxScaler()
if all_train_dfs:
    global_scaler.fit(pd.concat(all_train_dfs))
else:
    raise ValueError("未发现有效数据集。")

# ========== 初始化全局变量 ==========
window_size = 16
horizon_step = 1  # 预测 30 分钟后
convergence_mse, convergence_r2, generation_params = [], [], []
best_mse, best_r2, best_ind = float('inf'), -float('inf'), None
fitness_cache = {}


# ========== 适应度评估函数 (修正信息泄露) ==========
def eval_params(ind):
    global best_mse, best_r2, best_ind
    alpha, beta = ind

    cache_key = (alpha, beta)
    if cache_key in fitness_cache:
        return fitness_cache[cache_key]

    print(f"！ 正在评估个体: alpha={alpha}, beta={beta}")

    if alpha < 200 or alpha > 450 or beta < 600 or beta > 1200 or (beta - alpha) < 100:
        return (float('inf'),)

    ms, rs, val_ms_norm = [], [], []
    patient_weights = []

    # 遍历字典，确保数据在时序上的物理隔离
    for pid, df_p in patient_data_dict.items():
        L = len(df_p)

        if L < alpha:
            model_obj = build_frozen_lstm(len(features))
        elif L > beta:
            model_obj = LSTMModel(len(features))
        else:
            model_obj = GRUModel(len(features))

        # 必须同时减去预测步长，否则后面会越界
        L_X_raw = L - window_size - horizon_step
        if L_X_raw <= 15: continue

        train_split = int(0.7 * L_X_raw)
        val_split = int(0.85 * L_X_raw)

        if train_split < 8 or (val_split - train_split) < 2 or (L_X_raw - val_split) < 2:
            continue

        # 使用全局统一基准，严禁在循环内重新 fit
        arr = global_scaler.transform(df_p)

        X, y = [], []
        # 🚨 修正越界 Bug：循环必须减去 horizon_step
        for i in range(len(arr) - window_size - horizon_step):
            X.append(arr[i:i + window_size])
            y.append(arr[i + window_size + horizon_step, 0])
        X, y = np.array(X), np.array(y)

        X_tr, y_tr = X[:train_split], y[:train_split]
        X_val, y_val = X[train_split:val_split], y[train_split:val_split]
        X_te, y_te = X[val_split:], y[val_split:]

        pred_val, pred_te = train_and_predict_torch(model_obj, X_tr, y_tr, X_val, y_val, X_te)

        # 验证集 MSE 用于演化决策
        val_mse_norm = mean_squared_error(y_val, pred_val)
        val_ms_norm.append(val_mse_norm)
        patient_weights.append(len(y_val))

        def inverse_target(y_arr):
            dummy = np.zeros((len(y_arr), len(features)))
            dummy[:, 0] = y_arr.flatten()
            return global_scaler.inverse_transform(dummy)[:, 0]

        y_te_orig = inverse_target(y_te)
        pred_te_orig = inverse_target(pred_te)
        mse = mean_squared_error(y_te_orig, pred_te_orig)
        r2 = r2_score(y_te_orig, pred_te_orig)
        ms.append(mse)
        rs.append(r2)

    if not ms or not val_ms_norm:
        fitness_cache[cache_key] = (float('inf'),)
        return (float('inf'),)

    # 加权平均
    avg_val_m_norm = np.average(val_ms_norm, weights=patient_weights)
    avg_te_m, avg_te_r = np.mean(ms), np.mean(rs)

    # 保持原 Excel 记录逻辑
    generation_params.append({"Alpha": alpha, "Beta": beta, "MSE": avg_te_m, "R2": avg_te_r})

    # 【修正】最优决策必须基于验证集，严禁偷看测试集
    if avg_val_m_norm < best_mse:
        best_mse, best_r2, best_ind = avg_val_m_norm, avg_te_r, ind[:]

    print(f"√ 评估完: Val_MSE_W_Norm={avg_val_m_norm:.6f} | Te_MSE={avg_te_m:.4f}, Te_R2={avg_te_r:.4f}")

    fitness_cache[cache_key] = (avg_val_m_norm,)
    return (avg_val_m_norm,)


# ========== 遗传算法设置 (保持原逻辑) ==========
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


def gen_indiv():
    a = random.randint(200, 450)
    b = random.randint(max(a + 50, 600), 1200)
    return creator.Individual([a, b])


def custom_mutate(ind):
    if random.random() < 0.3:
        ind[0] = random.randint(200, 450)
    if random.random() < 0.3:
        ind[1] = random.randint(600, 1200)
    if ind[0] >= ind[1]:
        ind[1] = ind[0] + 50
    return ind,


toolbox = base.Toolbox()
toolbox.register("individual", gen_indiv)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", custom_mutate)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", eval_params)


# ========== 精英交叉策略主循环 ==========
# ========== 精英交叉策略主循环 (修复为真·精英绝对保留) ==========
def eaEliteCross(pop, toolbox, cxpb, mutpb_init, ngen, elite=1):
    gene_var_ref = 3000
    min_mutpb, max_mutpb = 0.01, 0.5
    gene_var_history, mutpb_history = [], []
    mutpb = mutpb_init

    fits = list(map(toolbox.evaluate, pop))
    for ind, f in zip(pop, fits): ind.fitness.values = f

    for gen in range(1, ngen + 1):
        print(f"\n--- 第 {gen} 代进化 ---")

        # 1. 【绝对保护】：提取精英，放入保险箱，绝不参与后续变异
        elites = list(map(toolbox.clone, tools.selBest(pop, elite)))

        # 2. 【生成平民】：从原种群选出参与繁衍的父本
        offspring = list(map(toolbox.clone, toolbox.select(pop, len(pop) - elite)))

        # 动态变异率计算 (基于总体基因多样性)
        genes = np.array(pop)  # 多样性基于整个父代评估
        gene_var = np.var(genes[:, 0]) + np.var(genes[:, 1])
        gene_var_history.append(gene_var)
        mutpb = max(min_mutpb, min(mutpb_init * (gene_var_ref / (gene_var + 1e-6)), max_mutpb))
        mutpb_history.append(mutpb)

        # 3. 【平民繁衍】：仅仅对 offspring 进行交叉和变异
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(c1, c2)
                del c1.fitness.values, c2.fitness.values

        for ind in offspring:
            if random.random() < mutpb:
                toolbox.mutate(ind)
                del ind.fitness.values

        # 防止近亲繁殖导致死锁 (仅对 offspring 查重)
        from collections import Counter
        count = Counter(tuple(ind) for ind in offspring)
        repeated = [t for t, c in count.items() if c >= 3]
        for ind in offspring:
            if tuple(ind) in repeated and random.random() < 0.5:
                ind[0] = max(200, min(450, ind[0] + random.choice([-10, -5, 5, 10])))
                ind[1] = max(ind[0] + 50, min(1200, ind[1] + random.choice([-10, -5, 5, 10])))
                del ind.fitness.values

        # 4. 【平民评估】：评估那些基因发生改变的个体
        invalid = [i for i in offspring if not i.fitness.valid]
        for ind in invalid: ind.fitness.values = toolbox.evaluate(ind)

        # 5. 【新老交替】：新一代种群 = 原汁原味的绝对精英 + 进化后的平民
        pop[:] = elites + offspring

        # 打印当前代最强者 (因为 elites 在里面，所以只会越来越强，绝不退步)
        current_best = tools.selBest(pop, 1)[0]
        print(f"-> 第 {gen} 代最优验证 MSE (W_Norm): {current_best.fitness.values[0]:.6f}")

        # 记录用于画图的收敛曲线
        for entry in reversed(generation_params):
            if entry["Alpha"] == current_best[0] and entry["Beta"] == current_best[1]:
                convergence_mse.append(entry["MSE"])
                convergence_r2.append(entry["R2"])
                break

    df_var_mut = pd.DataFrame({
        "Generation": range(1, ngen + 1),
        "Diversity": gene_var_history,
        "MutationRate": mutpb_history
    })
    df_var_mut.to_excel(r"./diversity_mutation.xlsx", index=False)
    return pop, gene_var_history, mutpb_history


# ========== 执行主流程 ==========
NGEN = 20
POP_SIZE = 40

population = toolbox.population(n=POP_SIZE)
final_pop, var_hist, mut_hist = eaEliteCross(
    population,
    toolbox,
    cxpb=0.7,
    mutpb_init=0.3,
    ngen=NGEN,
    elite=1
)

# ========== 汇总与可视化 (保持原结构) ==========
print(f"！最优参数组合： {best_ind}")
sorted_list = sorted(generation_params, key=lambda x: x['MSE'])
print("√ 最优10个参数组合（R2 与 MSE 排行）---精英交叉策略：")
for i, v in enumerate(sorted_list[:10], 1):
    print(f"排名 {i}: 最优α = {v['Alpha']}, 最优β = {v['Beta']}, R2 = {v['R2']:.4f}, MSE = {v['MSE']:.4f}")

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

end_time = time.time()
elapsed = end_time - start_time
print(f"time： 本次运行耗时{time.strftime('%H:%M:%S', time.gmtime(elapsed))}")