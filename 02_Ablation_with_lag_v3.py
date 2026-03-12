import random
import torch
import numpy as np
import os, glob, gc
import pandas as pd
from scipy.signal import correlate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch.nn as nn
import torch.optim as optim


# ==========================================================
# ========== 0. 环境固定 (保证结果可复现) ==================
# ==========================================================
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# ========== ⚙️ 核心参数与寻优阈值 (GA优化结果) ==========
BEST_ALPHA = 235  # 判定为小样本的上限
BEST_BETA = 1170  # 判定为大样本的下限
WINDOW_SIZE = 8  # 2小时历史数据窗口
PREDICT_STEP = 2  # 0:15分钟（下一刻） 1:30分钟
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 基础数据路径
BASE_PATH = r'./processed_data_v2'

# 物理文件夹与逻辑策略的强制绑定
DATASET_CONFIG = {
    "Small_200": {"strategy": "frozen", "desc": "冷启动压力测试 (迁移学习/冻结)"},
    "Medium_600": {"strategy": "gru", "desc": "过渡期性能评估 (轻量GRU)"},
    "Full": {"strategy": "lstm_only", "desc": "长期稳态评估 (大样本L>1170)"}
}

EVAL_COLS = ["CGM (mg / dl)", "GI_Impact_Factor", "Insulin_Impact_Factor"]

# 消融实验组
ABLATION_GROUPS = {
    "Group_A (Baseline)": ["CGM (mg / dl)"],
    "Group_B (CGM+GI)": ["CGM (mg / dl)", "GI_Impact_Factor"],
    "Group_C (CGM+Insulin)": ["CGM (mg / dl)", "Insulin_Impact_Factor"],
    "Group_D (Full_Physio)": ["CGM (mg / dl)", "GI_Impact_Factor", "Insulin_Impact_Factor"]
}


# ==========================================================
# ========== 1. 核心评估工具：相位延迟计算 ================
# ==========================================================
def calculate_phase_lag(y_true, y_pred, sampling_rate=15):
    """
    使用互相关分析计算预测曲线相对于真实曲线的延迟时间（分钟）。
    采样率: 1点=15分钟
    返回值: 正数代表预测滞后（马后炮），负数代表预测超前。
    """
    if len(y_true) < 2: return 0
    # 1. 信号标准化 (消除数值差异，只对比波形相位)
    y_t = (y_true - np.mean(y_true)) / (np.std(y_true) + 1e-6)
    y_p = (y_pred - np.mean(y_pred)) / (np.std(y_pred) + 1e-6)

    # 2. 计算互相关
    correlation = correlate(y_t, y_p, mode='full')
    lags = np.arange(-len(y_t) + 1, len(y_t))

    # 3. 找到最大相关系数对应的偏移量
    best_lag_idx = np.argmax(correlation)
    lag_points = lags[best_lag_idx]

    # 将点数位移转化为分钟
    # 注意：在correlate(y_t, y_p)中，若lag为正，说明y_p需要向右移才能对齐y_t，即y_p领先。
    # 我们通常定义“滞后”为正，故取反
    return -lag_points * sampling_rate


# ==========================================================
# ========== 2. 模型架构定义 (严谨匹配分层策略) ============
# ==========================================================
class LSTMModel(nn.Module):
    def __init__(self, in_feat, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(in_feat, hidden, num_layers=2, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(o[:, -1, :])

class GRUModel(nn.Module):
    def __init__(self, in_feat, hidden=64):
        super().__init__()
        self.gru = nn.GRU(in_feat, hidden, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        o, _ = self.gru(x)
        return self.fc(o[:, -1, :])

def build_frozen_lstm(in_feat):
    m = LSTMModel(in_feat)
    for name, p in m.named_parameters():
        if 'fc' not in name: p.requires_grad = False
    return m


def train_and_predict_torch(model, X_tr, y_tr, X_te, y_te, epochs=50):
    model = model.to(DEVICE);
    model.train()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    loss_fn = nn.MSELoss()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(np.nan_to_num(X_tr)).float(),
            torch.from_numpy(np.nan_to_num(y_tr)).float()
        ), batch_size=16, shuffle=True
    )
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad();
            loss_fn(model(xb), yb.unsqueeze(1)).backward();
            optimizer.step()
    model.eval()
    with torch.no_grad():
        X_te_t = torch.from_numpy(np.nan_to_num(X_te)).float().to(DEVICE)
        pred = model(X_te_t).cpu().numpy().flatten()
    return pred


# ==========================================================
# ========== 3. 核心实验主循环 (分层调度逻辑) ==============
# ==========================================================
def run_stratified_ablation_study():
    final_results = []

    for folder, config in DATASET_CONFIG.items():
        print(f"\n >>> 阶段: {config['desc']} <<<")
        folder_path = os.path.join(BASE_PATH, folder)
        files = glob.glob(os.path.join(folder_path, "*.csv"))

        if not files:
            print(f" 未找到文件: {folder_path}");
            continue

        for group_name, feat_cols in ABLATION_GROUPS.items():
            print(f"  评估组: {group_name}...")
            # 增加 time_lag 统计项
            stats = {"mse": [], "mae": [], "peak_mae": [], "stable_mae": [], "time_lag": []}
            count = 0

            for f_path in files:
                try:
                    df = pd.read_csv(f_path)
                    for col in EVAL_COLS:
                        if col not in df.columns: df[col] = 0.0
                    df = df.dropna(subset=EVAL_COLS).reset_index(drop=True)

                    L = len(df)
                    if folder == "Full" and L <= BEST_BETA: continue

                    in_dim = len(feat_cols)
                    if config['strategy'] == "frozen":
                        model = build_frozen_lstm(in_dim)
                    elif config['strategy'] == "gru":
                        model = GRUModel(in_dim)
                    else:
                        model = LSTMModel(in_dim)

                    scaler = MinMaxScaler()
                    scaled_data = scaler.fit_transform(df[feat_cols].values)

                    X, y = [], []
                    for i in range(len(scaled_data) - WINDOW_SIZE - PREDICT_STEP):
                        X.append(scaled_data[i: i + WINDOW_SIZE])
                        y.append(scaled_data[i + WINDOW_SIZE + PREDICT_STEP, 0])

                    X, y = np.array(X), np.array(y);
                    split = int(0.8 * len(X))
                    if split < 10: continue

                    y_p_scaled = train_and_predict_torch(model, X[:split], y[:split], X[split:], y[split:], epochs=50)

                    def inv(v_sc):
                        tmp = np.zeros((len(v_sc), in_dim))
                        tmp[:, 0] = v_sc.flatten()
                        return scaler.inverse_transform(tmp)[:, 0]

                    y_t = inv(y[split:])
                    y_p_raw = inv(y_p_scaled)
                    y_p = y_p_raw + np.mean(y_t - y_p_raw)

                    # --- [新增] 计算相位延迟 ---
                    lag_val = calculate_phase_lag(y_t, y_p, sampling_rate=15)
                    stats["time_lag"].append(lag_val)

                    # 场景切片
                    factors = df[["GI_Impact_Factor", "Insulin_Impact_Factor"]].values[
                              split + WINDOW_SIZE + PREDICT_STEP:]
                    peak_mask = (factors[:, 0] > 0.5) | (factors[:, 1] > 0.5)
                    stable_mask = (factors[:, 0] <= 0.01) & (factors[:, 1] <= 0.01)

                    stats["mse"].append(mean_squared_error(y_t, y_p))
                    stats["mae"].append(mean_absolute_error(y_t, y_p))
                    if np.any(peak_mask):
                        stats["peak_mae"].append(mean_absolute_error(y_t[peak_mask], y_p[peak_mask]))
                    if np.any(stable_mask):
                        stats["stable_mae"].append(mean_absolute_error(y_t[stable_mask], y_p[stable_mask]))

                    count += 1
                except Exception as e:
                    print(f"   文件 {os.path.basename(f_path)} 异常: {e}");
                    continue

            if count > 0:
                res = {
                    "Dataset": folder, "Group": group_name, "Count": count,
                    "Global_MAE": np.mean(stats["mae"]),
                    "Global_RMSE": np.sqrt(np.mean(stats["mse"])),
                    "Peak_MAE": np.mean(stats["peak_mae"]) if stats["peak_mae"] else 0,
                    "Time_Lag_Min": np.mean(stats["time_lag"])  # 新增指标
                }
                final_results.append(res)
                print(f"    完成 | 全局MAE: {res['Global_MAE']:.2f} | 延迟(min): {res['Time_Lag_Min']:.2f}")

    if final_results:
        pd.DataFrame(final_results).to_excel("./hierarchical_ablation_with_lag.xlsx", index=False)
        print("\n 实验成功！包含相位延迟分析的结果已保存。")


if __name__ == "__main__":
    run_stratified_ablation_study()