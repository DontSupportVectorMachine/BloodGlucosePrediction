import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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


seed_everything(42)  # 固定种子，保证每次运行结果一致

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

WINDOW_SIZE = 16
HORIZON_STEP = 1  # 预测30分钟后
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
        # x shape: (batch, window_size, features)
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        delta = self.fc(out)  # 网络现在只需要专心学习 "波动变化量 (Delta)"

        # x[:, -1, 0] 是当前窗口最后一个时间点的真实已知血糖值 (归一化状态)
        last_known_cgm = x[:, -1, 0].unsqueeze(1)

        # 最终预测 = 当前已知血糖 + 模型预测的波动量
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
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.2)
        self.flatten_dim = 32 * (window_size // 2)
        self.fc = nn.Linear(self.flatten_dim, 1)

    def forward(self, x):
        # 保存原始 x，用于提取最后一个时间点的已知血糖
        last_known_cgm = x[:, -1, 0].unsqueeze(1)

        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.flatten(1)
        delta = self.fc(x)

        return last_known_cgm + delta


# =================  核心评价类 =================
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
            self.best_loss = val_loss;
            self.best_state = copy.deepcopy(model.state_dict())
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_loss = val_loss;
            self.best_state = copy.deepcopy(model.state_dict());
            self.counter = 0


#  【此处已替换为你的终极统一标准版 calculate_phase_lag】
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


def calculate_metrics(y_true, y_pred):
    try:
        r2 = r2_score(y_true, y_pred)
    except:
        r2 = 0
    return {"MAE": mean_absolute_error(y_true, y_pred), "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)), "R2": r2}


#  保持不动：严谨医学版 Clarke Error Grid 绘图函数
def plot_clarke_error_grid(ref_values, pred_values, title="Clarke Error Grid Analysis", safe_filename="CEG_Plot"):
    ref_values = np.array(ref_values).flatten()
    pred_values = np.array(pred_values).flatten()

    if len(ref_values) != len(pred_values):
        raise ValueError("Reference and predicted arrays must be the same length.")

    n = len(ref_values)
    total = np.zeros(5)

    for i in range(n):
        y, yp = ref_values[i], pred_values[i]
        if (yp <= 70 and y <= 70) or (0.8 * y <= yp <= 1.2 * y):
            total[0] += 1
        elif (y >= 180 and yp <= 70) or (y <= 70 and yp >= 180):
            total[4] += 1
        elif (70 <= y <= 290 and yp >= y + 110) or (130 <= y <= 180 and yp <= (7 / 5) * y - 182):
            total[2] += 1
        elif (y >= 240 and 70 <= yp <= 180) or (y <= 175 / 3 and 70 <= yp <= 180) or (
                175 / 3 <= y <= 70 and yp >= 1.2 * y):
            total[3] += 1
        else:
            total[1] += 1

    percentage = (total / n) * 100

    fig, ax = plt.subplots(figsize=(8, 8), dpi=300)
    ax.plot(ref_values, pred_values, 'ko', markersize=4, markerfacecolor='k', markeredgecolor='k', alpha=0.5)
    ax.set_xlim([0, 400]);
    ax.set_ylim([0, 400])
    ax.set_xlabel('Reference Blood Glucose (mg/dL)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Predicted Blood Glucose (mg/dL)', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
    ax.set_aspect('equal', adjustable='box')

    ax.plot([0, 400], [0, 400], 'k:', linewidth=1.5)
    ax.plot([0, 175 / 3], [70, 70], 'k-', linewidth=1.5);
    ax.plot([175 / 3, 400 / 1.2], [70, 400], 'k-', linewidth=1.5)
    ax.plot([70, 70], [84, 400], 'k-', linewidth=1.5);
    ax.plot([0, 70], [180, 180], 'k-', linewidth=1.5)
    ax.plot([70, 290], [180, 400], 'k-', linewidth=1.5);
    ax.plot([70, 70], [0, 56], 'k-', linewidth=1.5)
    ax.plot([70, 400], [56, 320], 'k-', linewidth=1.5);
    ax.plot([180, 180], [0, 70], 'k-', linewidth=1.5)
    ax.plot([180, 400], [70, 70], 'k-', linewidth=1.5);
    ax.plot([240, 240], [70, 180], 'k-', linewidth=1.5)
    ax.plot([240, 400], [180, 180], 'k-', linewidth=1.5);
    ax.plot([130, 180], [0, 70], 'k-', linewidth=1.5)

    font_opts = {'fontsize': 16, 'fontweight': 'bold', 'color': 'black'}
    ax.text(30, 20, 'A', **font_opts);
    ax.text(30, 150, 'D', **font_opts);
    ax.text(30, 380, 'E', **font_opts)
    ax.text(150, 380, 'C', **font_opts);
    ax.text(160, 20, 'C', **font_opts);
    ax.text(380, 20, 'E', **font_opts)
    ax.text(380, 120, 'D', **font_opts);
    ax.text(380, 260, 'B', **font_opts);
    ax.text(280, 380, 'B', **font_opts)

    stats_text = (f"Zone A: {percentage[0]:.1f}%\nZone B: {percentage[1]:.1f}%\nZone C: {percentage[2]:.1f}%\n"
                  f"Zone D: {percentage[3]:.1f}%\nZone E: {percentage[4]:.1f}%\n--------------------\n"
                  f"Safe (A+B): {percentage[0] + percentage[1]:.1f}%")
    bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.95)
    ax.text(390, 10, stats_text, fontsize=11, family='monospace', verticalalignment='bottom',
            horizontalalignment='right', bbox=bbox_props)

    plt.tight_layout()
    plt.savefig(f"./{safe_filename}.png", dpi=300, bbox_inches='tight')
    plt.close()


#  修复早停偷看漏洞：内部切分 15% 验证集
def train_eval_pytorch(model, X_tr_all, y_tr_all, X_test_raw):
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    split_v = int(len(X_tr_all) * 0.85)
    X_tr, y_tr = X_tr_all[:split_v], y_tr_all[:split_v]
    X_val, y_val = X_tr_all[split_v:], y_tr_all[split_v:]

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).float()),
                              batch_size=BATCH_SIZE, shuffle=True)
    X_val_t = torch.from_numpy(X_val).float().to(DEVICE)
    y_val_t = torch.from_numpy(y_val).float().to(DEVICE)

    early_stop = EarlyStopping(patience=5)

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            v_loss = criterion(model(X_val_t), y_val_t)
            early_stop(v_loss.item(), model)
            if early_stop.early_stop: break

    if early_stop.best_state is not None: model.load_state_dict(early_stop.best_state)
    model.eval()
    with torch.no_grad():
        X_te_t = torch.from_numpy(X_test_raw).float().to(DEVICE)
        return model(X_te_t).cpu().numpy()


#  修复预训练重叠隐患：严格划分全局前20患者作为预训练群
def run_pretraining_for_all_modes():
    print("\n [阶段 0] 正在预训练通用基座 (严格隔离测试患者群)...")
    full_path = os.path.join(DATA_BASE_DIR, "Full")
    # 只取排序后的前20个文件作为预训练群
    files = sorted(glob(os.path.join(full_path, "*.csv")))[:20]
    pretrained_dict = {}

    for mode_name, feats in FEATURE_MODES.items():
        print(f"   -> 预训练抽取: {mode_name}")
        all_X, all_y = [], []
        for f in files:
            df = pd.read_csv(f).dropna(subset=["CGM (mg / dl)", "GI_Impact_Factor", "Insulin_Impact_Factor"])
            if len(df) < WINDOW_SIZE + HORIZON_STEP + 10: continue

            raw_vals = df[feats].values
            split_idx = int(len(raw_vals) * 0.8)
            scaler = MinMaxScaler()
            scaler.fit(raw_vals[:split_idx])
            data = scaler.transform(raw_vals)

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

    ceg_data = {
        "SVR": {"true": [], "pred": []},
        "1D-CNN": {"true": [], "pred": []},
        "Proposed (Ours)": {"true": [], "pred": []}
    }

    for mode_name, current_features in FEATURE_MODES.items():
        print(f"\n" + "=" * 60)
        print(f" 当前特征赛道: {mode_name} | 包含特征: {current_features}")
        print("=" * 60)

        for exp_key, config in EXPERIMENTS.items():
            print(f"\n 启动实验: {config['desc']}")
            all_files = sorted(glob(os.path.join(DATA_BASE_DIR, config['folder'], "*.csv")))
            #  隔离修复：测试评价阶段跳过前20，仅在 [20:40] 的独立未见患者上评价泛化性能
            test_files = all_files[20:100]

            for model_name in MODELS_TO_COMPARE:
                print(f"  运行模型: {model_name}...")
                p_metrics = {"MAE": [], "RMSE": [], "R2": [], "Lag": [], "Peak_MAE": [], "Peak_Lag": []}

                for f_path in test_files:
                    try:
                        df = pd.read_csv(f_path).dropna(
                            subset=["CGM (mg / dl)", "GI_Impact_Factor", "Insulin_Impact_Factor"])
                        L = len(df)
                        if config['folder'] == "Full" and L <= BEST_BETA: continue
                        if L < WINDOW_SIZE + HORIZON_STEP + 10: continue

                        raw_vals = df[current_features].values
                        split_idx = int(len(raw_vals) * 0.8)

                        scaler = MinMaxScaler()
                        scaler.fit(raw_vals[:split_idx])
                        data_scaled = scaler.transform(raw_vals)

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

                        seq_split = int(len(X) * 0.8)
                        X_train, X_test = X[:seq_split], X[seq_split:]
                        y_train, y_test = y[:seq_split], y[seq_split:]
                        peak_test = peak_flags[seq_split:]

                        input_dim = len(current_features)
                        if model_name == "SVR":
                            svr = SVR(kernel='rbf', C=100)
                            y_pred = svr.fit(X_train.reshape(len(X_train), -1), y_train.ravel()).predict(
                                X_test.reshape(len(X_test), -1)).reshape(-1, 1)
                        elif model_name == "1D-CNN":
                            cnn = CNNModel(input_dim, WINDOW_SIZE)
                            y_pred = train_eval_pytorch(cnn, X_train, y_train, X_test)
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
                            y_pred = train_eval_pytorch(model, X_train, y_train, X_test)


                        def inv_trans(val):
                            temp = np.zeros((len(val), input_dim))
                            temp[:, 0] = val.flatten()
                            return scaler.inverse_transform(temp)[:, 0]


                        y_t_real, y_p_real = inv_trans(y_test), inv_trans(y_pred)

                        # 在测试未知患者的稳态/冷启动/过渡期搜集CEG点阵
                        if mode_name == "3D_Full_Physio" and config['folder'] == "Full":
                            ceg_data[model_name]["true"].extend(y_t_real.flatten())
                            ceg_data[model_name]["pred"].extend(y_p_real.flatten())

                        m = calculate_metrics(y_t_real, y_p_real)

                        #  这里调用统一后的 calculate_phase_lag
                        lag = calculate_phase_lag(y_t_real, y_p_real)

                        p_metrics["MAE"].append(m["MAE"]);
                        p_metrics["RMSE"].append(m["RMSE"]);
                        p_metrics["R2"].append(m["R2"]);
                        p_metrics["Lag"].append(lag)

                        if np.any(peak_test):
                            m_p = calculate_metrics(y_t_real[peak_test], y_p_real[peak_test])

                            #  这里也调用统一后的 calculate_phase_lag
                            lag_p = calculate_phase_lag(y_t_real[peak_test], y_p_real[peak_test])
                            p_metrics["Peak_MAE"].append(m_p["MAE"]);
                            p_metrics["Peak_Lag"].append(lag_p)

                    except Exception as e:
                        continue

                if p_metrics["MAE"]:
                    res = {
                        "Feature_Mode": mode_name, "Dataset": config['desc'], "Model": model_name,
                        "MAE_Global": np.mean(p_metrics["MAE"]), "R2_Global": np.mean(p_metrics["R2"]),
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

    print("\n" + "=" * 100)
    print("。。。 正在基于真实的、患者级隔离环境 生成三大模型的临床安全性评价图 (Clarke Error Grid)...")
    for m_name in MODELS_TO_COMPARE:
        t_vals = ceg_data[m_name]["true"]
        p_vals = ceg_data[m_name]["pred"]
        if len(t_vals) > 0:
            safe_name = m_name.replace(" (Ours)", "").replace("-", "_").replace(" ", "_")
            file_name = f"CEG_{safe_name}_3D_Full_Stage"
            plot_clarke_error_grid(t_vals, p_vals, title=f"{m_name} (3D) in Full Stage", safe_filename=file_name)