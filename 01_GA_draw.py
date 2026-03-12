# GA Result Visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import griddata

# plt.style.use('ggplot')
plt.rcParams['figure.facecolor'] = 'white'  # 图片背景全白
plt.rcParams['axes.facecolor'] = 'white'    # 坐标系背景全白
plt.rcParams['grid.color'] = '#e0e0e0'      # 浅灰色网格线
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['axes.unicode_minus'] = False





# Load result file (modify path if necessary)
df = pd.read_excel(r"./ga_results_torch.xlsx")
df.columns = df.columns.str.strip()
# 检查 Generation 列存不存在，如果不存在或变成了索引，就强行修复
if 'Generation' not in df.columns:
    if df.index.name == 'Generation':
        df = df.reset_index() # 如果是索引，降级为普通列
    else:
        df['Generation'] = range(1, len(df) + 1) # 如果真没有，按行号生成 1, 2, 3...

# -----------------------------------
# Figure 1: Fitness Evolution Trend (MSE)
# -----------------------------------
best_fits = df['MSE'].cummin()
mean_fits = df['MSE'].expanding().mean()
worst_fits = df['MSE'].cummax()

plt.figure(figsize=(12, 6))
plt.plot(best_fits, 'o--', color='royalblue', label='Best Fitness (Lowest MSE)')
plt.plot(mean_fits, 'x--', color='darkorange', label='Average Fitness')
plt.plot(worst_fits, '*--', color='forestgreen', label='Worst Fitness (Highest MSE)')
plt.xlabel("Individual Index", fontsize=12)
plt.ylabel("MSE Value", fontsize=12)
plt.title("Figure 1: GA Fitness Evolution (with Elitism)", fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("Figure1_Fitness_Evolution.png", dpi=300) # 保存图一
plt.show()

# -----------------------------------
# Figure 2: 3D Search Space Surface & Trajectory
# -----------------------------------

alpha_range = np.linspace(df["Alpha"].min(), df["Alpha"].max(), 50)
beta_range = np.linspace(df["Beta"].min(), df["Beta"].max(), 50)
Alpha_grid, Beta_grid = np.meshgrid(alpha_range, beta_range)
points = df[["Alpha", "Beta"]].values
values = df["MSE"].values

# 【核心修复】：将 cubic 修改为 linear，消除边缘过冲产生的深蓝色“假坑”
MSE_grid = griddata(points, values, (Alpha_grid, Beta_grid), method='linear')
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d', facecolor='white')
surf = ax.plot_surface(Alpha_grid, Beta_grid, MSE_grid, cmap=cm.viridis, edgecolor='none', alpha=0.6)
ax.scatter(df["Alpha"], df["Beta"], df["MSE"], c='red', s=30, edgecolors='black', label='Search Points')
# Set viewing angle: elev (elevation), azim (azimuth)
ax.view_init(elev=35, azim=145)
best_idx = df['MSE'].idxmin()
ax.text(df.loc[best_idx, 'Alpha'], df.loc[best_idx, 'Beta'], df.loc[best_idx, 'MSE'] + 2,
        "Optimal Solution", fontsize=11, color='darkred', fontweight='bold') # 加粗最优解文字
# 透明度 alpha 从 0.9 降到 0.6，让红点透出来

cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
cbar.set_label('MSE Value', fontsize=11)
ax.set_xlabel("Alpha", labelpad=10, fontsize=12)
ax.set_ylabel("Beta", labelpad=10, fontsize=12)
ax.set_zlabel("MSE", labelpad=10, fontsize=12)
ax.set_title("Figure 2: Search Space Surface and Trajectory (Alpha-Beta-MSE)", fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig("Figure2_Search_Space.png", dpi=300) # 保存图二
plt.show()

# -----------------------------------
# Figure 3: Evaluation Metrics Trend per Generation (MSE & R²) [Dual Y-axis]
# -----------------------------------
fig, ax1 = plt.subplots(figsize=(12, 6))

# Left Y-axis: MSE
color1 = 'tomato'
ax1.set_xlabel("Generation", fontsize=12)
ax1.set_ylabel("Mean Squared Error (MSE)", fontsize=12, color=color1)
ax1.plot(df['Generation'], df['MSE'], 'o--', color=color1, label='MSE')
ax1.tick_params(axis='y', labelcolor=color1)

# Right Y-axis: R²
ax2 = ax1.twinx()
color2 = 'royalblue'
ax2.set_ylabel("Coefficient of Determination (R²)", fontsize=12, color=color2)
ax2.plot(df['Generation'], df['R2'], 's--', color=color2, label='R²')
ax2.tick_params(axis='y', labelcolor=color2)

# Title and layout
plt.title("Figure 3: Evaluation Metrics Trend per Generation", fontsize=14, fontweight='bold')
fig.tight_layout()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("Figure3_Metrics_Trend.png", dpi=300) # 保存图三
plt.show()

# -----------------------------------
# Figure 4: Diversity & Mutation Rate
# -----------------------------------
# Reload file for the second dataset
file_path = r"./diversity_mutation.xlsx"
df_div = pd.read_excel(file_path)

# Dual-axis plot
fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.set_xlabel("Generation", fontsize=12)
ax1.set_ylabel("Diversity", color='green', fontsize=12)
ax1.plot(df_div["Generation"], df_div["Diversity"], marker='o', color='green', label="Diversity")
ax1.tick_params(axis='y', labelcolor='green')

ax2 = ax1.twinx()
ax2.set_ylabel("Mutation Rate", color='blue', fontsize=12)
ax2.plot(df_div["Generation"], df_div["MutationRate"], marker='x', linestyle='--', color='blue', label="Mutation Rate")
ax2.tick_params(axis='y', labelcolor='blue')

plt.title("Figure 4: Diversity and Mutation Rate Trends across Generations", fontsize=14, fontweight='bold')
fig.tight_layout()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("Figure4_Diversity_Mutation.png", dpi=300) # 保存图四
plt.show()

# -----------------------------------
# Figure 5: Comprehensive Convergence Trend (MSE & R²)
# -----------------------------------
fig, ax1 = plt.subplots(figsize=(12, 6))

# 1. 提取图一的核心：MSE 的最好、平均、最差轨迹
best_mse = df['MSE'].cummin()
mean_mse = df['MSE'].expanding().mean()
worst_mse = df['MSE'].cummax()

# 2. 提取原图五的核心：R² 的历史最优轨迹
best_r2 = df['R2'].cummax()

# ---------- 左侧 Y 轴：绘制 MSE (完全看齐图一的样式) ----------
color_best = 'royalblue'
color_mean = 'darkorange'
color_worst = 'forestgreen'

ax1.set_xlabel("Individual Evaluations (Timeline)", fontsize=12)
ax1.set_ylabel("Mean Squared Error (MSE)", fontsize=12)

# 加入 markersize=4 减小点的大小，防止 280 个点挤在一起变成粗线
ax1.plot(df['Generation'], best_mse, 'o--', color=color_best, label='Best MSE (Lowest)', markersize=4, alpha=0.8)
ax1.plot(df['Generation'], mean_mse, 'x--', color=color_mean, label='Average MSE', markersize=4, alpha=0.8)
ax1.plot(df['Generation'], worst_mse, '*--', color=color_worst, label='Worst MSE (Highest)', markersize=4, alpha=0.8)
ax1.tick_params(axis='y')

# ---------- 右侧 Y 轴：绘制 R² ----------
ax2 = ax1.twinx()
color_r2 = 'crimson' # 使用深红色以区别于左侧的颜色
ax2.set_ylabel("Coefficient of Determination (R²)", fontsize=12, color=color_r2)
# 使用方形标记和实线突出显示 R² 的攀升
ax2.plot(df['Generation'], best_r2, 's-', linewidth=2, color=color_r2, label='Best-so-far R²', markersize=4, alpha=0.9)
ax2.tick_params(axis='y', labelcolor=color_r2)

# ---------- 图标题与布局 ----------
plt.title("Figure 5: Comprehensive Evolution Trend of Optimization Process", fontsize=14, fontweight='bold')
plt.grid(True, linestyle='--', alpha=0.6)

# 巧妙合并左右两个 Y 轴的图例，并放在中右侧空白处
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')

fig.tight_layout()
plt.savefig("Figure5_Comprehensive_Convergence.png", dpi=300) # 自动保存高清大图
plt.show()