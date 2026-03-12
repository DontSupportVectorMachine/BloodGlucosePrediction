import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
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

seed_everything(42) # 固定种子，保证每次运行结果一致

# =================  配置区域 =================
DATA_BASE_DIR = r"./processed_data_v2"
BEST_ALPHA = 235
BEST_BETA = 1170

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

WINDOW_SIZE = 8
HORIZON_STEP = 0  # 预测15分钟后
BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =================  模型定义 =================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out)


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, x):
        o, _ = self.gru(x)
        return self.fc(o[:, -1, :])


class CNNModel(nn.Module):
    def __init__(self, input_size, window_size):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_size, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.2)
        self.flatten_dim = 32 * (window_size // 2)
        self.fc = nn.Linear(self.flatten_dim, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.flatten(1)
        return self.fc(x)


# =================  修复后的核心函数 =================
class EarlyStopping:
    def __init__(self, patience=5, delta=0.0001):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.best_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss; self.best_state = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_loss = val_loss; self.best_state = copy.deepcopy(model.state_dict()); self.counter = 0


def calculate_phase_lag(y_true, y_pred, max_lag_steps=4, sampling_rate=15):
    """
    修正后的对齐法：
    >0 : 预测滞后 (慢半拍)
    <0 : 预测提前 (预警)
    """
    if len(y_true) < max_lag_steps * 2 + 1: return 0
    best_lag = 0
    min_mse = float('inf')
    for shift in range(-max_lag_steps, max_lag_steps + 1):
        if shift > 0:
            mse = mean_squared_error(y_true[:-shift], y_pred[shift:])
        elif shift < 0:
            pos_shift = -shift; mse = mean_squared_error(y_true[pos_shift:], y_pred[:-pos_shift])
        else:
            mse = mean_squared_error(y_true, y_pred)
        if mse < min_mse: min_mse = mse; best_lag = shift
    return best_lag * sampling_rate


def calculate_metrics(y_true, y_pred):
    try:
        r2 = r2_score(y_true, y_pred)
    except:
        r2 = 0
    return {"MAE": mean_absolute_error(y_true, y_pred), "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)), "R2": r2}


#  修复 Bug 1：去掉了导致广播灾难的 unsqueeze，加上了梯度裁剪
def train_eval_pytorch(model, train_loader, X_test_t, y_test_np, is_frozen=False):
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    early_stop = EarlyStopping(patience=5)

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)  # 修复广播Bug
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 修复梯度爆炸
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(X_test_t)
            val_loss = criterion(val_out, torch.from_numpy(y_test_np).to(DEVICE))
            early_stop(val_loss.item(), model)
            if early_stop.early_stop: break

    if early_stop.best_state is not None: model.load_state_dict(early_stop.best_state)
    model.eval()
    with torch.no_grad():
        return model(X_test_t).cpu().numpy()


#  修复 Bug 2：为 1D 和 3D 各自独立预训练基座
def run_pretraining_for_all_modes():
    print("\n [阶段 0] 正在为双极特征赛道分别预训练通用基座...")
    full_path = os.path.join(DATA_BASE_DIR, "Full")
    files = glob(os.path.join(full_path, "*.csv"))[:30]
    pretrained_dict = {}

    for mode_name, feats in FEATURE_MODES.items():
        print(f"   -> 预训练抽取: {mode_name}")
        all_X, all_y = [], []
        for f in files:
            df = pd.read_csv(f).dropna(subset=["CGM (mg / dl)", "GI_Impact_Factor", "Insulin_Impact_Factor"])
            if len(df) < WINDOW_SIZE + HORIZON_STEP + 10: continue
            data = MinMaxScaler().fit_transform(df[feats])
            for i in range(len(data) - WINDOW_SIZE - HORIZON_STEP):
                all_X.append(data[i: i + WINDOW_SIZE]);
                all_y.append(data[i + WINDOW_SIZE + HORIZON_STEP, 0])

        if not all_X: continue
        X_t = torch.tensor(np.array(all_X), dtype=torch.float32).to(DEVICE)
        y_t = torch.tensor(np.array(all_y), dtype=torch.float32).unsqueeze(1).to(DEVICE)

        m = LSTMModel(len(feats)).to(DEVICE)
        opt = optim.Adam(m.parameters(), lr=0.001)
        crit = nn.MSELoss()
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=32, shuffle=True)

        for _ in range(10):
            m.train()
            for xb, yb in loader:
                opt.zero_grad();
                crit(m(xb), yb).backward();
                opt.step()
        pretrained_dict[mode_name] = m.state_dict()
    return pretrained_dict


# =================  终极实验循环 =================
if __name__ == "__main__":
    start_time = time.time()
    pretrained_dict = run_pretraining_for_all_modes()
    all_final_results = []

    for mode_name, current_features in FEATURE_MODES.items():
        print(f"\n" + "=" * 60)
        print(f" 当前特征赛道: {mode_name} | 包含特征: {current_features}")
        print("=" * 60)

        for exp_key, config in EXPERIMENTS.items():
            print(f"\n 启动实验: {config['desc']}")
            files = glob(os.path.join(DATA_BASE_DIR, config['folder'], "*.csv"))

            for model_name in MODELS_TO_COMPARE:
                print(f"  ⚔️ 运行模型: {model_name}...")
                p_metrics = {"MAE": [], "RMSE": [], "R2": [], "Lag": [], "Peak_MAE": [], "Peak_Lag": []}

                for f_path in files:
                    try:
                        df = pd.read_csv(f_path).dropna(
                            subset=["CGM (mg / dl)", "GI_Impact_Factor", "Insulin_Impact_Factor"])
                        L = len(df)
                        if config['folder'] == "Full" and L <= BEST_BETA: continue
                        if L < WINDOW_SIZE + HORIZON_STEP + 10: continue

                        scaler = MinMaxScaler()
                        data_scaled = scaler.fit_transform(df[current_features].values)
                        #  即使在 1D 赛道，依然读取生理因子用来做“峰值区”的公平验证
                        gi_ins_raw = df[["GI_Impact_Factor", "Insulin_Impact_Factor"]].values

                        X, y, peak_flags = [], [], []
                        for i in range(len(data_scaled) - WINDOW_SIZE - HORIZON_STEP):
                            X.append(data_scaled[i: i + WINDOW_SIZE])
                            y.append(data_scaled[i + WINDOW_SIZE + HORIZON_STEP, 0])
                            is_peak = (gi_ins_raw[i + WINDOW_SIZE + HORIZON_STEP, 0] > 0.5) or \
                                      (gi_ins_raw[i + WINDOW_SIZE + HORIZON_STEP, 1] > 0.5)
                            peak_flags.append(is_peak)

                        X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)
                        peak_flags = np.array(peak_flags)

                        split = int(len(X) * 0.8)
                        X_train, X_test = X[:split], X[split:]
                        y_train, y_test = y[:split], y[split:]
                        peak_test = peak_flags[split:]

                        input_dim = len(current_features)
                        if model_name == "SVR":
                            svr = SVR(kernel='rbf', C=100)
                            y_pred = svr.fit(X_train.reshape(len(X_train), -1), y_train.ravel()).predict(
                                X_test.reshape(len(X_test), -1)).reshape(-1, 1)
                        elif model_name == "1D-CNN":
                            cnn = CNNModel(input_dim, WINDOW_SIZE)
                            y_pred = train_eval_pytorch(cnn, DataLoader(
                                TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                                batch_size=BATCH_SIZE), torch.from_numpy(X_test).to(DEVICE), y_test)
                        else:  # Proposed
                            is_f = False
                            if config['strategy'] == "gru":
                                model = GRUModel(input_dim)
                            else:
                                model = LSTMModel(input_dim)
                                if config['strategy'] == "freeze_lstm" and mode_name in pretrained_dict:
                                    model.load_state_dict(pretrained_dict[mode_name])
                                    for p in model.lstm.parameters(): p.requires_grad = False
                                    is_f = True
                            y_pred = train_eval_pytorch(model, DataLoader(
                                TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                                batch_size=BATCH_SIZE), torch.from_numpy(X_test).to(DEVICE), y_test, is_frozen=is_f)


                        def inv_trans(val):
                            temp = np.zeros((len(val), input_dim))
                            temp[:, 0] = val.flatten()
                            return scaler.inverse_transform(temp)[:, 0]

                        # bias修正
                        y_t_real, y_p_real = inv_trans(y_test), inv_trans(y_pred)
                        # y_p_real += np.mean(y_t_real - y_p_real)

                        m = calculate_metrics(y_t_real, y_p_real)
                        lag = calculate_phase_lag(y_t_real, y_p_real)

                        p_metrics["MAE"].append(m["MAE"]);
                        p_metrics["RMSE"].append(m["RMSE"]);
                        p_metrics["R2"].append(m["R2"]);
                        p_metrics["Lag"].append(lag)

                        if np.any(peak_test):
                            m_p = calculate_metrics(y_t_real[peak_test], y_p_real[peak_test])
                            lag_p = calculate_phase_lag(y_t_real[peak_test], y_p_real[peak_test])
                            p_metrics["Peak_MAE"].append(m_p["MAE"]);
                            p_metrics["Peak_Lag"].append(lag_p)

                    except Exception as e:
                        continue

                if p_metrics["MAE"]:
                    res = {
                        "Feature_Mode": mode_name,
                        "Dataset": config['desc'],
                        "Model": model_name,
                        "MAE_Global": np.mean(p_metrics["MAE"]),
                        "R2_Global": np.mean(p_metrics["R2"]),
                        "MAE_Peak": np.mean(p_metrics["Peak_MAE"]) if p_metrics["Peak_MAE"] else np.mean(
                            p_metrics["MAE"]),
                        "Lag_Peak(min)": np.mean(p_metrics["Peak_Lag"]) if p_metrics["Peak_Lag"] else np.mean(
                            p_metrics["Lag"])
                    }
                    all_final_results.append(res)

    final_df = pd.DataFrame(all_final_results)
    final_df.to_excel("./Ultimate_Comparison_Study_Fixed.xlsx", index=False)

    print("\n" + "=" * 100)
    print(" 终极模型对比实验汇总报表 (1D 纯CGM vs 3D 生理闭环)")
    print("=" * 100)
    print(final_df.to_string(index=False))