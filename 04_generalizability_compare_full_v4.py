import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from glob import glob
import os
from scipy.stats import gamma
import warnings
import copy, random

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
TRAIN_DATA_DIR = r".\processed_data_v2\Full"
OHIO_TEST_DIR = r".\ohiot1dm-glucose-dataset-main\Ohio Data\Ohio2020_processed\test"

# 【新增】GA 寻优得到的最新动态阈值
BEST_ALPHA = 231
BEST_BETA = 838

FEATURE_COLS = ["CGM (mg / dl)", "GI_Impact_Factor", "Insulin_Impact_Factor"]
MODELS_TO_COMPARE = ["SVR", "1D-CNN", "Proposed (Ours)"]

WINDOW_SIZE = 16
HORIZON_STEP = 1
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001
KERNEL_STEPS = 48
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f" 运行设备: {DEVICE} | 零样本泛化测试：已激活 Alpha({BEST_ALPHA}) & Beta({BEST_BETA}) 动态分层调度")


# =================  生理动力学核函数 =================
def get_gamma_kernel():
    t_gi = np.linspace(0, KERNEL_STEPS - 1, KERNEL_STEPS)
    kernel_gi = gamma.pdf(t_gi, a=3, scale=2)
    return kernel_gi / np.max(kernel_gi)


def get_insulin_kernel():
    t = np.linspace(0, KERNEL_STEPS - 1, KERNEL_STEPS)
    tau_short = 3.5
    k = (t / tau_short) * np.exp(1 - t / tau_short)
    return k / np.max(k)


GAMMA_KERNEL = get_gamma_kernel()
INSULIN_KERNEL = get_insulin_kernel()


# =================  严谨的 Ohio 数据适配器 =================
def load_and_adapt_ohio_data(file_path):
    try:
        df_raw = pd.read_csv(file_path)
        if 'cbg' not in df_raw.columns: return None

        df_raw['group_id'] = df_raw.index // 3
        df_15min = df_raw.groupby('group_id').agg({
            'cbg': 'first',
            'carbInput': 'sum',
            'bolus': 'sum',
            'basal': 'mean'
        }).reset_index(drop=True)

        df_15min['CGM (mg / dl)'] = df_15min['cbg']

        carb_vals = df_15min['carbInput'].fillna(0).values
        carb_max = carb_vals.max()
        normalized_carb = carb_vals / carb_max if carb_max > 0 else np.zeros(len(df_15min))
        df_15min['GI_Impact_Factor'] = np.convolve(normalized_carb, GAMMA_KERNEL, mode='full')[:len(df_15min)]

        bolus_vals = df_15min['bolus'].fillna(0).values
        if 'basal' in df_15min.columns and not df_15min['basal'].isna().all():
            basal_vals = df_15min['basal'].fillna(method='ffill').fillna(method='bfill').values
            basal_mean = np.mean(basal_vals)
            basal_delta_dose = (basal_vals - basal_mean) * 0.25
            total_insulin_dose = np.maximum(bolus_vals + basal_delta_dose, 0)
        else:
            total_insulin_dose = bolus_vals

        dose_max = total_insulin_dose.max()
        normalized_insulin = total_insulin_dose / dose_max if dose_max > 0 else np.zeros(len(df_15min))
        df_15min['Insulin_Impact_Factor'] = np.convolve(normalized_insulin, INSULIN_KERNEL, mode='full')[:len(df_15min)]

        df_final = df_15min[FEATURE_COLS].dropna()
        if len(df_final) < WINDOW_SIZE + HORIZON_STEP + 5: return None
        return df_final
    except Exception as e:
        return None


# ================= 📊 评估工具 =================
def calculate_metrics(y_true, y_pred):
    try:
        r2 = r2_score(y_true, y_pred)
    except:
        r2 = 0
    return {"MAE": mean_absolute_error(y_true, y_pred), "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)), "R2": r2}


def calculate_phase_lag(y_true, y_pred, max_lag_steps=4, sampling_rate=15):
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


# ================= 🏗️ 模型定义 =================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.3)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        delta = self.fc(out)  # 残差波动预测
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
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(0.2)
        # 动态推断维度
        dummy = self.pool(self.relu(self.conv1(torch.zeros(1, input_size, window_size))))
        self.flatten_dim = dummy.view(1, -1).size(1)
        self.fc = nn.Linear(self.flatten_dim, 1)

    def forward(self, x):
        last_known_cgm = x[:, -1, 0].unsqueeze(1)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.flatten(1)
        delta = self.fc(x)
        return last_known_cgm + delta


def train_eval_pytorch(model, train_loader, X_test_t, is_frozen=False):
    model.to(DEVICE)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    with torch.no_grad():
        return model(X_test_t).cpu().numpy().flatten()


# ================= 主程序 =================
def run_generalization_test():
    print(f"\n 阶段一: 加载源域 (T2DM) 提取跨病种生理规律...")
    train_files = glob(os.path.join(TRAIN_DATA_DIR, "*.csv"))[:30]

    all_X, all_y = [], []
    for f in train_files:
        df = pd.read_csv(f).dropna(subset=FEATURE_COLS)
        raw_vals = df[FEATURE_COLS].values
        split_idx = int(len(raw_vals) * 0.8)
        scaler = MinMaxScaler()
        scaler.fit(raw_vals[:split_idx])
        data = scaler.transform(raw_vals)

        for i in range(len(data) - WINDOW_SIZE - HORIZON_STEP):
            all_X.append(data[i: i + WINDOW_SIZE]);
            all_y.append(data[i + WINDOW_SIZE + HORIZON_STEP, 0])

    X_train_np = np.array(all_X, dtype=np.float32)
    y_train_np = np.array(all_y, dtype=np.float32).reshape(-1, 1)
    X_train_t = torch.tensor(X_train_np).to(DEVICE)
    y_train_t = torch.tensor(y_train_np).to(DEVICE)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True)

    base_lstm = LSTMModel(3).to(DEVICE)
    opt_base = optim.Adam(base_lstm.parameters(), lr=LEARNING_RATE)
    crit = nn.MSELoss()
    for _ in range(20):
        base_lstm.train()
        for xb, yb in train_loader:
            opt_base.zero_grad();
            crit(base_lstm(xb), yb).backward();
            opt_base.step()

    pretrained_lstm_state = copy.deepcopy(base_lstm.state_dict())
    print("   T2DM Base LSTM 提取完毕。")

    print(f"\n阶段二: 模拟 Ohio T1DM 目标域自适应测试...")
    ohio_files = glob(os.path.join(OHIO_TEST_DIR, "*.csv"))
    if not ohio_files: print("❌ 未找到 Ohio 测试文件！"); return

    #  EVAL_STAGES 不再包含策略，策略现在是【动态触发】的
    EVAL_STAGES = {
        "Small_200": {"slice": 200},
        "Medium_600": {"slice": 600},
        "Full": {"slice": None}
    }

    all_results = []

    for stage_name, stage_config in EVAL_STAGES.items():
        print(f"\n📍 正在模拟阶段: {stage_name}")

        for model_name in MODELS_TO_COMPARE:
            p_metrics = {"MAE": [], "R2": [], "Peak_MAE": [], "Peak_RMSE": [], "Peak_Lag": []}

            for file_path in ohio_files:
                df_ohio = load_and_adapt_ohio_data(file_path)
                if df_ohio is None: continue
                df_current = df_ohio.iloc[:stage_config["slice"]] if stage_config["slice"] else df_ohio
                if len(df_current) < WINDOW_SIZE + HORIZON_STEP + 10: continue

                raw_ohio = df_current[FEATURE_COLS].values
                split_ohio = int(len(raw_ohio) * 0.8)
                scaler = MinMaxScaler();
                scaler.fit(raw_ohio[:split_ohio])
                X_data = scaler.transform(raw_ohio)

                gi_ins_raw = df_current[["GI_Impact_Factor", "Insulin_Impact_Factor"]].values

                X_seq, y_seq, peak_flags = [], [], []
                for i in range(len(X_data) - WINDOW_SIZE - HORIZON_STEP):
                    X_seq.append(X_data[i: i + WINDOW_SIZE])
                    y_seq.append(X_data[i + WINDOW_SIZE + HORIZON_STEP, 0])
                    target_idx = i + WINDOW_SIZE + HORIZON_STEP
                    is_peak = (gi_ins_raw[target_idx, 0] > 0.1) or (gi_ins_raw[target_idx, 1] > 0.1)
                    peak_flags.append(is_peak)

                X_np = np.array(X_seq, dtype=np.float32);
                y_np = np.array(y_seq, dtype=np.float32).reshape(-1, 1)
                split = int(len(X_np) * 0.8);
                X_train, X_test = X_np[:split], X_np[split:]
                y_train, y_true_scaled = y_np[:split], y_np[split:];
                peak_test = np.array(peak_flags)[split:]

                X_train_t = torch.tensor(X_train).to(DEVICE);
                y_train_t = torch.tensor(y_train).to(DEVICE)
                loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=False)
                X_test_t = torch.tensor(X_test).to(DEVICE)

                if model_name == "SVR":
                    svr = SVR(kernel='rbf', C=100)
                    y_pred_scaled = svr.fit(X_train.reshape(len(X_train), -1), y_train.ravel()).predict(
                        X_test.reshape(len(X_test), -1)).flatten()
                elif model_name == "1D-CNN":
                    model = CNNModel(3, WINDOW_SIZE)
                    y_pred_scaled = train_eval_pytorch(model, loader, X_test_t)
                else:  # Proposed (Ours) 核心调度逻辑 🚀
                    L_logic = len(df_current)
                    if L_logic < BEST_ALPHA:  # 触发冷启动：冻结基座
                        model = LSTMModel(3);
                        model.load_state_dict(pretrained_lstm_state, strict=False)
                        for p in model.lstm.parameters(): p.requires_grad = False
                    elif L_logic > BEST_BETA:  # 长期稳态：全量学习
                        model = LSTMModel(3)
                    else:  # 过渡期：轻量GRU
                        model = GRUModel(3)

                    y_pred_scaled = train_eval_pytorch(model, loader, X_test_t)

                def inv(v):
                    tmp = np.zeros((len(v), 3));
                    tmp[:, 0] = v.flatten();
                    return scaler.inverse_transform(tmp)[:, 0]

                y_pred, y_actual = inv(y_pred_scaled), inv(y_true_scaled.flatten())
                m_all = calculate_metrics(y_actual, y_pred)

                if np.any(peak_test):
                    m_peak = calculate_metrics(y_actual[peak_test], y_pred[peak_test])
                    lag_peak = calculate_phase_lag(y_actual[peak_test], y_pred[peak_test])
                else:
                    m_peak = m_all;
                    lag_peak = calculate_phase_lag(y_actual, y_pred)

                p_metrics["MAE"].append(m_all["MAE"]);
                p_metrics["R2"].append(m_all["R2"])
                p_metrics["Peak_MAE"].append(m_peak["MAE"]);
                p_metrics["Peak_RMSE"].append(m_peak["RMSE"]);
                p_metrics["Peak_Lag"].append(lag_peak)

            if p_metrics["MAE"]:
                all_results.append({
                    "阶段": stage_name, "模型": model_name,
                    "Global MAE": np.mean(p_metrics["MAE"]), "Global R²": np.mean(p_metrics["R2"]),
                    "Peak MAE": np.mean(p_metrics["Peak_MAE"]), "Peak Lag (min)": np.mean(p_metrics["Peak_Lag"])
                })

    df_res = pd.DataFrame(all_results)
    print("\n" + "=" * 90 + "\n 表 5-6  Ohio T1DM 跨病种泛化性能终极报表 (已启用动态分层架构)\n" + "=" * 90)
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    run_generalization_test()