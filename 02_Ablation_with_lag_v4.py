import random, re, glob, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim
from scipy.signal import correlate
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import copy, gc


# ========== 0. 环境固定 ==========
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)
# ==========  核心参数 ==========
BEST_ALPHA = 231
BEST_BETA = 838
WINDOW_SIZE = 16  # 4小时生理窗口
PREDICT_STEP = 1  # 预测下一节点 (30分钟超前预测)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_PATH = r'./processed_data_v2'

# 核心策略配置
DATASET_CONFIG = {
    "Full": {"strategy": "lstm_only", "desc": "阶段1: 长期稳态评估 (用于生成全局迁移权重)"},
    "Small_200": {"strategy": "frozen", "desc": "阶段2: 冷启动压力测试 (基于最佳Full权重)"},
    "Medium_600": {"strategy": "gru", "desc": "阶段3: 过渡期性能评估 (轻量GRU)"}
}

EVAL_COLS = ["CGM (mg / dl)", "GI_Impact_Factor", "Insulin_Impact_Factor"]
ABLATION_GROUPS = {
    "Group_A (Baseline)": ["CGM (mg / dl)"],
    "Group_B (CGM+GI)": ["CGM (mg / dl)", "GI_Impact_Factor"],
    "Group_C (CGM+Insulin)": ["CGM (mg / dl)", "Insulin_Impact_Factor"],
    "Group_D (Full_Physio)": ["CGM (mg / dl)", "GI_Impact_Factor", "Insulin_Impact_Factor"]
}

# 全局最优权重存储桶
GLOBAL_BEST_MODELS = {gn: {"weights": None, "val_loss": float('inf')} for gn in ABLATION_GROUPS}


# ========== 1. 工具类 ==========
class EarlyStopping:
    def __init__(self, patience=5, delta=0.0001):
        self.patience, self.delta = patience, delta
        self.best_loss, self.counter, self.early_stop, self.best_state = None, 0, False, None

    def __call__(self, val_loss, model):
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss, self.counter = val_loss, 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True


def calculate_phase_lag(y_true, y_pred, max_lag_steps=4, sampling_rate=15):
    """
    基于临床严格对齐的 MSE 局部寻优法：
    >0 : 预测滞后 (慢半拍)
    <0 : 预测提前 (预警)
    """
    if len(y_true) < max_lag_steps * 2 + 1: return 0
    best_lag = 0
    min_mse = float('inf')

    # 限制寻优范围在 [-60分钟, +60分钟]，防止幽灵对齐
    for shift in range(-max_lag_steps, max_lag_steps + 1):
        if shift > 0:
            mse = mean_squared_error(y_true[:-shift], y_pred[shift:])
        elif shift < 0:
            pos_shift = -shift
            mse = mean_squared_error(y_true[pos_shift:], y_pred[:-pos_shift])
        else:
            mse = mean_squared_error(y_true, y_pred)

        if mse < min_mse:
            min_mse = mse
            best_lag = shift

    return best_lag * sampling_rate


# ========== 2. 模型定义 (升级为残差架构) ==========
class LSTMModel(nn.Module):
    def __init__(self, in_feat, hidden=64):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(in_feat, hidden, num_layers=2, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        # x shape: (batch, window_size, features)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        delta = self.fc(out)  # 预测波动量

        # 提取当前窗口最后一刻真实值 (第0列 CGM)
        last_known_cgm = x[:, -1, 0].unsqueeze(1)
        return last_known_cgm + delta


class GRUModel(nn.Module):
    def __init__(self, in_feat, hidden=64):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(in_feat, hidden, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        o, _ = self.gru(x)
        delta = self.fc(o[:, -1, :])
        last_known_cgm = x[:, -1, 0].unsqueeze(1)
        return last_known_cgm + delta


def build_frozen_lstm(in_feat, group_name):
    m = LSTMModel(in_feat)
    if GLOBAL_BEST_MODELS[group_name]["weights"] is not None:
        m.load_state_dict(GLOBAL_BEST_MODELS[group_name]["weights"])
        for name, p in m.named_parameters():
            if 'fc' not in name: p.requires_grad = False
    return m


def train_and_predict_torch(model, X_tr_raw, y_tr_raw, X_te, epochs=50):
    model = model.to(DEVICE)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    loss_fn = nn.MSELoss()

    n_samples = len(X_tr_raw)
    use_early_stop = n_samples > 20
    split_v = int(n_samples * 0.85) if use_early_stop else n_samples

    X_tr, y_tr = X_tr_raw[:split_v], y_tr_raw[:split_v]
    X_val, y_val = X_tr_raw[split_v:], y_tr_raw[split_v:]

    loader_tr = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float()),
        batch_size=min(len(X_tr), 16), shuffle=True
    )

    early_stop = EarlyStopping(patience=5)

    for _ in range(epochs):
        model.train()
        for xb, yb in loader_tr:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss_fn(model(xb), yb.unsqueeze(1)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if use_early_stop:
            model.eval()
            with torch.no_grad():
                X_val_t = torch.from_numpy(X_val).float().to(DEVICE)
                y_val_t = torch.from_numpy(y_val).float().to(DEVICE)
                v_loss = loss_fn(model(X_val_t), y_val_t.unsqueeze(1)).item()
                early_stop(v_loss, model)
                if early_stop.early_stop: break

    if use_early_stop and early_stop.best_state:
        model.load_state_dict(early_stop.best_state)

    model.eval()
    with torch.no_grad():
        X_te_t = torch.from_numpy(np.nan_to_num(X_te)).float().to(DEVICE)
        pred = model(X_te_t).cpu().numpy().flatten()

    final_v_loss = early_stop.best_loss if use_early_stop else 999.0
    gc.collect();
    torch.cuda.empty_cache()
    return pred, final_v_loss


# ========== 3. 主循环 ==========
def run_stratified_ablation_study():
    final_results = []
    for folder, config in DATASET_CONFIG.items():
        print(f"\n >>> {config['desc']} <<<")
        folder_path = os.path.join(BASE_PATH, folder)
        files = glob.glob(os.path.join(folder_path, "*.csv"))
        if not files: continue

        for group_name, feat_cols in ABLATION_GROUPS.items():
            print(f"  评估组: {group_name}...")
            stats = {"mse": [], "mae": [], "peak_mae": [], "time_lag": []}
            count = 0

            for f_path in files:
                try:
                    df_raw = pd.read_csv(f_path, parse_dates=['Date']).sort_values('Date').set_index('Date')
                    df = df_raw.resample('15min').asfreq()

                    for col in EVAL_COLS:
                        if col not in df.columns: df[col] = 0.0
                    df[EVAL_COLS] = df[EVAL_COLS].ffill().fillna(0.0)

                    if folder == "Full" and len(df) <= BEST_BETA: continue

                    raw_full = df[EVAL_COLS].values
                    split_idx_raw = int(len(raw_full) * 0.8)
                    scaler = MinMaxScaler().fit(raw_full[:split_idx_raw])
                    scaled_full = scaler.transform(raw_full)

                    feat_indices = [EVAL_COLS.index(c) for c in feat_cols]
                    scaled_data, cgm_target_scaled = scaled_full[:, feat_indices], scaled_full[:, 0]

                    X, y, y_orig_idx = [], [], []
                    for i in range(len(scaled_data) - WINDOW_SIZE - PREDICT_STEP):
                        X.append(scaled_data[i: i + WINDOW_SIZE])
                        target_idx = i + WINDOW_SIZE + PREDICT_STEP
                        y.append(cgm_target_scaled[target_idx])
                        y_orig_idx.append(target_idx)

                    if not X: continue
                    X, y, y_orig_idx = np.array(X), np.array(y), np.array(y_orig_idx)
                    split = int(0.8 * len(X))
                    buffer_size = WINDOW_SIZE + PREDICT_STEP + 1
                    if split < (buffer_size + 5): continue

                    X_train, y_train = X[:split - buffer_size], y[:split - buffer_size]
                    X_test, y_test = X[split:], y[split:]
                    y_test_orig_idx = y_orig_idx[split:]

                    if config['strategy'] == "frozen":
                        model = build_frozen_lstm(len(feat_cols), group_name)
                    elif config['strategy'] == "gru":
                        model = GRUModel(len(feat_cols))
                    else:
                        model = LSTMModel(len(feat_cols))

                    y_p_scaled, v_loss = train_and_predict_torch(model, X_train, y_train, X_test)

                    if folder == "Full" and config['strategy'] == "lstm_only":
                        if v_loss < GLOBAL_BEST_MODELS[group_name]["val_loss"]:
                            GLOBAL_BEST_MODELS[group_name]["val_loss"] = v_loss
                            GLOBAL_BEST_MODELS[group_name]["weights"] = copy.deepcopy(model.state_dict())

                    def inv_cgm(v_sc):
                        dummy = np.zeros((len(v_sc), len(EVAL_COLS)))
                        dummy[:, 0] = v_sc.flatten()
                        return scaler.inverse_transform(dummy)[:, 0]

                    y_t, y_p = inv_cgm(y_test), inv_cgm(y_p_scaled)

                    # 统一调用 MSE 局部寻优法
                    stats["time_lag"].append(calculate_phase_lag(y_t, y_p))

                    peak_mask = []
                    for idx in y_test_orig_idx:
                        impact_area = df.iloc[max(0, idx - 4): idx][["GI_Impact_Factor", "Insulin_Impact_Factor"]]
                        peak_mask.append(impact_area.max().max() > 0.5)
                    peak_mask = np.array(peak_mask)

                    stats["mse"].append(mean_squared_error(y_t, y_p))
                    stats["mae"].append(mean_absolute_error(y_t, y_p))
                    if np.any(peak_mask):
                        stats["peak_mae"].append(mean_absolute_error(y_t[peak_mask], y_p[peak_mask]))
                    count += 1
                except Exception as e:
                    print(f"    文件 {os.path.basename(f_path)} 异常: {e}");
                    continue

            if count > 0:
                res = {"Dataset": folder, "Group": group_name, "Count": count,
                       "Global_MAE": np.mean(stats["mae"]), "Global_RMSE": np.sqrt(np.mean(stats["mse"])),
                       "Peak_MAE": np.mean(stats["peak_mae"]) if stats["peak_mae"] else np.mean(stats["mae"]),
                       "Time_Lag_Min": np.mean(stats["time_lag"])}
                final_results.append(res)
                print(f"  完成 | MAE: {res['Global_MAE']:.2f} | 延迟: {res['Time_Lag_Min']:.2f}min")

    pd.DataFrame(final_results).to_excel("./hierarchical_ablation_final.xlsx", index=False)
    print("\n 实验成功！包含残差修正与严谨评价标尺的结果已保存。")


if __name__ == "__main__":
    run_stratified_ablation_study()