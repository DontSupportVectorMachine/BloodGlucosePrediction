# ==========================================================
# ========== 修正核心：1. 杜绝 0 值逻辑 (ffill + bfill) =======
# ========== 2. 增强 Peak_MAE 评价的生理覆盖范围 ============
# ========== 3. 优化 SVR 与 NN 的多维特征无泄露对齐 =========
# ==========================================================

import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from glob import glob
import os
import warnings
import copy
import time
import random
import torch

warnings.filterwarnings('ignore')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)  # 固定种子，保证每次运行结果一致

# =================  配置区域 =================
DATA_BASE_DIR = r"./processed_data_v2"
BEST_ALPHA = 231
BEST_BETA = 838

EXPERIMENTS = {
    "Small_Exp": {"folder": "Small_200", "strategy": "freeze_lstm", "desc": "冷启动阶段 (Small_200)"},
    "Medium_Exp": {"folder": "Medium_600", "strategy": "gru", "desc": "过渡阶段 (Medium_600)"},
    "Full_Exp": {"folder": "Full", "strategy": "full_lstm", "desc": "长期稳态阶段 (Full_Large)"}
}

FEATURE_MODES = {
    "1D_CGM_Only": ["CGM (mg / dl)"],
    "3D_Full_Physio": ["CGM (mg / dl)", "GI_Impact_Factor", "Insulin_Impact_Factor"]
}

MODELS_TO_COMPARE = ["SVR", "1D-CNN", "Proposed (Ours)"]

WINDOW_SIZE = 16  # 已统一 16
HORIZON_STEP = 1  # 已统一 30分钟 (t+2)
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =================  预处理工具 =================
def safe_preprocess(df, feats):
    """杜绝生理学 0 值，确保因果填充"""
    df = df.resample('15min').asfreq()
    for col in feats:
        if col not in df.columns: df[col] = np.nan
    df[feats] = df[feats].interpolate(method='linear').ffill().bfill()
    return df


# =================  模型定义 (残差架构) =================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        delta = self.fc(self.dropout(out[:, -1, :]))
        last_known_cgm = x[:, -1, 0].unsqueeze(1)
        return last_known_cgm + delta


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        o, _ = self.gru(x)
        delta = self.fc(o[:, -1, :])
        last_known_cgm = x[:, -1, 0].unsqueeze(1)
        return last_known_cgm + delta


class CNNModel(nn.Module):
    def __init__(self, input_size, window_size):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU();
        self.pool = nn.MaxPool1d(kernel_size=2);
        self.dropout = nn.Dropout(0.2)

        # 修正：采用动态维度推断，移除硬编码
        dummy_input = torch.zeros(1, input_size, window_size)
        dummy_out = self.pool(self.relu(self.conv1(dummy_input)))
        self.flatten_dim = dummy_out.view(1, -1).size(1)
        self.fc = nn.Linear(self.flatten_dim, 1)

    def forward(self, x):
        last_val = x[:, -1, 0].unsqueeze(1)
        x = x.permute(0, 2, 1)
        x = self.dropout(self.pool(self.relu(self.conv1(x)))).flatten(1)
        return last_val + self.fc(x)


# =================  核心评价类 (终极统一) =================
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
    """基于临床严格对齐的 MSE 局部寻优法"""
    if len(y_true) < max_lag_steps * 2 + 1: return 0
    best_lag, min_mse = 0, float('inf')
    for shift in range(-max_lag_steps, max_lag_steps + 1):
        if shift > 0:
            mse = mean_squared_error(y_true[:-shift], y_pred[shift:])
        elif shift < 0:
            pos = -shift; mse = mean_squared_error(y_true[pos:], y_pred[:-pos])
        else:
            mse = mean_squared_error(y_true, y_pred)
        if mse < min_mse: min_mse, best_lag = mse, shift
    return best_lag * sampling_rate


def calculate_metrics(y_true, y_pred):
    try:
        r2 = r2_score(y_true, y_pred)
    except:
        r2 = 0
    return {"MAE": mean_absolute_error(y_true, y_pred), "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)), "R2": r2}


def train_eval_pytorch(model, X_tr_all, y_tr_all, X_test_raw):
    model.to(DEVICE);
    criterion = nn.MSELoss();
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    split_v = int(len(X_tr_all) * 0.85)
    X_tr, y_tr, X_val, y_val = X_tr_all[:split_v], y_tr_all[:split_v], X_tr_all[split_v:], y_tr_all[split_v:]
    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float()),
                        batch_size=BATCH_SIZE, shuffle=True)
    X_v_t, y_v_t = torch.from_numpy(X_val).float().to(DEVICE), torch.from_numpy(y_val).float().to(DEVICE)
    early_stop = EarlyStopping(patience=5)
    for _ in range(EPOCHS):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE);
            opt.zero_grad();
            criterion(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0);
            opt.step()
        model.eval()
        with torch.no_grad():
            v_loss = criterion(model(X_v_t), y_v_t);
            early_stop(v_loss.item(), model)
            if early_stop.early_stop: break
    if early_stop.best_state: model.load_state_dict(early_stop.best_state)
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(X_test_raw).float().to(DEVICE)).cpu().numpy()


# =================  预训练基座生成 =================
def run_pretraining_for_all_modes():
    print("\n [阶段 0] 正在预训练基座 (已杜绝测试段泄漏)...")
    full_path = os.path.join(DATA_BASE_DIR, "Full")
    files = sorted(glob(os.path.join(full_path, "*.csv")))[:50]
    pretrained_dict = {}
    for mode_name, feats in FEATURE_MODES.items():
        all_X, all_y = [], []
        for f in files:
            try:
                df = safe_preprocess(pd.read_csv(f, parse_dates=['Date']).sort_values('Date').set_index('Date'), feats)
                raw = df[feats].values;
                cut = int(len(raw) * 0.8);
                sc = MinMaxScaler().fit(raw[:cut]);
                data = sc.transform(raw)
                for i in range(len(data) - WINDOW_SIZE - HORIZON_STEP):
                    all_X.append(data[i: i + WINDOW_SIZE]);
                    all_y.append(data[i + WINDOW_SIZE + HORIZON_STEP, 0])
            except:
                continue
        if not all_X: continue
        X_t, y_t = torch.tensor(np.array(all_X), dtype=torch.float32).to(DEVICE), torch.tensor(np.array(all_y),
                                                                                               dtype=torch.float32).reshape(
            -1, 1).to(DEVICE)
        m_lstm = LSTMModel(len(feats)).to(DEVICE);
        m_cnn = CNNModel(len(feats), WINDOW_SIZE).to(DEVICE)
        opt_l, opt_c = optim.Adam(m_lstm.parameters(), lr=1e-3), optim.Adam(m_cnn.parameters(), lr=1e-3);
        crit = nn.MSELoss()
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=32, shuffle=True)
        for _ in range(10):
            m_lstm.train();
            m_cnn.train()
            for xb, yb in loader:
                opt_l.zero_grad();
                crit(m_lstm(xb), yb).backward();
                opt_l.step()
                opt_c.zero_grad();
                crit(m_cnn(xb), yb).backward();
                opt_c.step()
        pretrained_dict[mode_name] = {'Proposed': copy.deepcopy(m_lstm.state_dict()),
                                      '1D-CNN': copy.deepcopy(m_cnn.state_dict())}
    return pretrained_dict


# =================  终极实验循环 =================
if __name__ == "__main__":
    start_time = time.time()
    pretrained_dict = run_pretraining_for_all_modes()
    all_results = []

    for mode_name, current_features in FEATURE_MODES.items():
        print(f"\n 特征赛道: {mode_name}")
        for exp_key, config in EXPERIMENTS.items():
            print(f" 实验: {config['desc']}")
            files = glob(os.path.join(DATA_BASE_DIR, config['folder'], "*.csv"))
            for model_name in MODELS_TO_COMPARE:
                p_metrics = {"MAE": [], "RMSE": [], "R2": [], "Lag": [], "Peak_MAE": [], "Weights": []}
                for f_path in files:
                    try:
                        df = safe_preprocess(
                            pd.read_csv(f_path, parse_dates=['Date']).sort_values('Date').set_index('Date'),
                            ["CGM (mg / dl)", "GI_Impact_Factor", "Insulin_Impact_Factor"])
                        if len(df) < WINDOW_SIZE + HORIZON_STEP + 15: continue
                        if config['folder'] == "Full" and len(df) <= BEST_BETA: continue

                        raw_f, raw_t = df[current_features].values, df[["CGM (mg / dl)"]].values
                        cut = int(len(raw_f) * 0.8);
                        sc_f = MinMaxScaler().fit(raw_f[:cut]);
                        sc_t = MinMaxScaler().fit(raw_t[:cut])
                        gi_raw = df["GI_Impact_Factor"].values

                        X, y, peak_flags = [], [], []
                        for i in range(len(df) - WINDOW_SIZE - HORIZON_STEP):
                            target_idx = i + WINDOW_SIZE + HORIZON_STEP
                            X.append(raw_f[i: i + WINDOW_SIZE]);
                            y.append(raw_t[target_idx, 0])
                            peak_flags.append(np.max(gi_raw[max(0, target_idx - 4):target_idx + 1]) > 0.5)

                        X, y, peak_flags = np.array(X), np.array(y).reshape(-1, 1), np.array(peak_flags)
                        split = int(len(X) * 0.8);
                        X_tr_raw, X_te_raw, y_tr_raw, y_te_raw, peak_te = X[:split], X[split:], y[:split], y[
                                                                                                           split:], peak_flags[
                                                                                                                    split:]

                        input_dim = len(current_features)
                        if model_name == "SVR":
                            svr_sc_f = StandardScaler().fit(raw_f[:cut]);
                            svr_sc_t = StandardScaler().fit(raw_t[:cut])
                            X_tr = svr_sc_f.transform(X_tr_raw.reshape(-1, input_dim)).reshape(len(X_tr_raw), -1)
                            X_te = svr_sc_f.transform(X_te_raw.reshape(-1, input_dim)).reshape(len(X_te_raw), -1)
                            y_pred_sc = SVR(kernel='rbf', C=100).fit(X_tr,
                                                                     svr_sc_t.transform(y_tr_raw).ravel()).predict(X_te)
                            y_p = svr_sc_t.inverse_transform(y_pred_sc.reshape(-1, 1)).flatten()
                        else:
                            X_tr = sc_f.transform(X_tr_raw.reshape(-1, input_dim)).reshape(len(X_tr_raw), WINDOW_SIZE,
                                                                                           -1)
                            X_te = sc_f.transform(X_te_raw.reshape(-1, input_dim)).reshape(len(X_te_raw), WINDOW_SIZE,
                                                                                           -1)
                            m = CNNModel(input_dim, WINDOW_SIZE) if model_name == "1D-CNN" else (
                                GRUModel(input_dim) if config['strategy'] == "gru" else LSTMModel(input_dim))
                            if config['strategy'] == "freeze_lstm" and mode_name in pretrained_dict and not isinstance(
                                    m, GRUModel):
                                m.load_state_dict(pretrained_dict[mode_name][
                                                      'Proposed' if model_name == "Proposed (Ours)" else '1D-CNN'],
                                                  strict=False)
                                if hasattr(m, 'lstm'):
                                    for p in m.lstm.parameters(): p.requires_grad = False
                            y_pred_sc = train_eval_pytorch(m, X_tr, sc_t.transform(y_tr_raw), X_te)
                            y_p = sc_t.inverse_transform(y_pred_sc.reshape(-1, 1)).flatten()

                        y_t = y_te_raw.flatten();
                        m_res = calculate_metrics(y_t, y_p);
                        lag = calculate_phase_lag(y_t, y_p)
                        p_metrics["MAE"].append(m_res["MAE"]);
                        p_metrics["RMSE"].append(m_res["RMSE"]);
                        p_metrics["R2"].append(m_res["R2"]);
                        p_metrics["Lag"].append(lag);
                        p_metrics["Weights"].append(len(y_t))
                        if np.any(peak_te): p_metrics["Peak_MAE"].append(
                            calculate_metrics(y_t[peak_te], y_p[peak_te])["MAE"])
                    except:
                        continue

                if p_metrics["MAE"]:
                    ws = p_metrics["Weights"]
                    all_results.append({
                        "Feature": mode_name, "Dataset": config['desc'], "Model": model_name,
                        "MAE_Global": np.average(p_metrics["MAE"], weights=ws),
                        "R2_Global": np.average(p_metrics["R2"], weights=ws),
                        "MAE_Peak": np.mean(p_metrics["Peak_MAE"]) if p_metrics["Peak_MAE"] else np.mean(
                            p_metrics["MAE"]),
                        "Lag(min)": np.mean(p_metrics["Lag"])
                    })
                    print(f"   √ [{model_name}] 测试完成。")

    pd.DataFrame(all_results).to_excel("./Ultimate_3D_Comparison_Study_Fixed.xlsx", index=False)
    print("\n 报表已生成至 Ultimate_3D_Comparison_Study_Fixed.xlsx")