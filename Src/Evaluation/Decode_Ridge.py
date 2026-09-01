from nlb_tools.nwb_interface import NWBDataset

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_squared_error
import joblib


# ============================================================
# 0. 参数设置
# ============================================================

# 修改成你自己的 NWB 文件路径
datapath = "/Users/heyunuo/Desktop/Intracortical Neural Decoding Project/Data/MC_Maze/sub-Jenkins_ses-full_desc-train_behavior+ecephys.nwb"

# bin 大小：20 ms
bin_size = 20

# 使用过去多少个 bin 的神经活动
# history_bins = 5 表示过去 5 * 20 ms = 100 ms
history_bins = 5

# Ridge 正则化强度
ridge_alpha = 100.0

# 是否只用部分数据测试代码
# 如果电脑运行慢，可以设成 True
use_subset = False

# 如果 use_subset=True，只使用前多少分钟数据
subset_minutes = 10

# 画图时显示多少个 bin
plot_len = 1000

# 结果保存路径
result_fig_path = "basic_ridge_decoding_result.png"
model_save_path = "basic_ridge_decoder.joblib"


# ============================================================
# 1. 读取 NWB 数据
# ============================================================

print("===== Loading dataset =====")

dataset = NWBDataset(datapath)

print("Dataset loaded")
print("dataset.data shape:", dataset.data.shape)

print("\nSignal types:")
print(dataset.data.columns.get_level_values("signal_type").unique())


# ============================================================
# 2. 提取神经活动和手部速度
# ============================================================

print("\n===== Extracting spikes and hand velocity =====")

# spikes: time x neurons
# hand_vel: time x 2
spikes_1ms = dataset.data["spikes"]
hand_vel_1ms = dataset.data["hand_vel"]

print("Original spikes_1ms shape:", spikes_1ms.shape)
print("Original hand_vel_1ms shape:", hand_vel_1ms.shape)

# ------------------------------------------------------------
# 处理 NaN
# ------------------------------------------------------------
# spikes 中的 NaN 填 0
# hand velocity 中的 NaN 用插值处理
spikes_1ms = spikes_1ms.fillna(0)
hand_vel_1ms = hand_vel_1ms.interpolate(limit_direction="both")

print("NaN in spikes_1ms after fill:", spikes_1ms.isna().sum().sum())
print("NaN in hand_vel_1ms after interpolate:", hand_vel_1ms.isna().sum().sum())


# ============================================================
# 3. 可选：只使用前几分钟数据
# ============================================================

if use_subset:
    print("\n===== Using subset of data =====")

    max_rows = subset_minutes * 60 * 1000

    spikes_1ms = spikes_1ms.iloc[:max_rows]
    hand_vel_1ms = hand_vel_1ms.iloc[:max_rows]

    print(f"Using first {subset_minutes} minutes")
    print("Subset spikes_1ms shape:", spikes_1ms.shape)
    print("Subset hand_vel_1ms shape:", hand_vel_1ms.shape)


# ============================================================
# 4. 重新 binning: 1 ms -> 20 ms
# ============================================================

print("\n===== Binning data =====")

n_time = len(spikes_1ms)
n_bins = n_time // bin_size

print("Original time points:", n_time)
print("Number of complete bins:", n_bins)

# 去掉不能整除 bin_size 的尾部
spikes_trim = spikes_1ms.iloc[:n_bins * bin_size]
hand_vel_trim = hand_vel_1ms.iloc[:n_bins * bin_size]

# 转成 numpy array
spikes_array = spikes_trim.values
hand_vel_array = hand_vel_trim.values

# spikes: 20 ms 内求和
# shape: n_bins x neurons
spikes_binned = spikes_array.reshape(n_bins, bin_size, -1).sum(axis=1)

# hand velocity: 20 ms 内求平均
# shape: n_bins x 2
hand_vel_binned = hand_vel_array.reshape(n_bins, bin_size, -1).mean(axis=1)

print("spikes_binned shape:", spikes_binned.shape)
print("hand_vel_binned shape:", hand_vel_binned.shape)

print("NaN in spikes_binned:", np.isnan(spikes_binned).sum())
print("NaN in hand_vel_binned:", np.isnan(hand_vel_binned).sum())


# ============================================================
# 5. 构造 spike history features
# ============================================================

print("\n===== Making lagged features =====")

def make_lagged_features(X, y, history_bins=5):
    """
    用过去 history_bins 个时间 bin 的神经活动预测当前 y。

    Parameters
    ----------
    X : array, shape = time x neurons
        binned spike counts

    y : array, shape = time x outputs
        hand velocity

    history_bins : int
        使用过去多少个 bin

    Returns
    -------
    X_lagged : array, shape = samples x neurons*history_bins
    y_lagged : array, shape = samples x outputs
    """

    X_list = []

    for t in range(history_bins, len(X)):
        # 取过去 history_bins 个 bin
        # 不包括当前 t
        window = X[t-history_bins:t, :]

        # flatten 成一维向量
        # shape: neurons * history_bins
        X_list.append(window.flatten())

    X_lagged = np.array(X_list)
    y_lagged = y[history_bins:, :]

    return X_lagged, y_lagged


X, y = make_lagged_features(
    spikes_binned,
    hand_vel_binned,
    history_bins=history_bins
)

print("X shape before NaN removal:", X.shape)
print("y shape before NaN removal:", y.shape)

print("NaN in X:", np.isnan(X).sum())
print("NaN in y:", np.isnan(y).sum())


# ============================================================
# 6. 删除仍然包含 NaN 或 inf 的样本
# ============================================================

print("\n===== Removing NaN / inf samples =====")

valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y).all(axis=1)

num_valid = valid_mask.sum()
num_invalid = len(valid_mask) - num_valid

print("Number of valid samples:", num_valid)
print("Number of invalid samples:", num_invalid)

X = X[valid_mask]
y = y[valid_mask]

print("X shape after NaN removal:", X.shape)
print("y shape after NaN removal:", y.shape)


# ============================================================
# 7. 划分训练集和测试集
# ============================================================

print("\n===== Splitting train and test data =====")

# 第一版使用最简单的时间顺序划分
# 前 80% 训练，后 20% 测试
n_samples = X.shape[0]
split_idx = int(n_samples * 0.8)

X_train = X[:split_idx]
y_train = y[:split_idx]

X_test = X[split_idx:]
y_test = y[split_idx:]

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# ============================================================
# 8. 定义并训练基础 Ridge decoder
# ============================================================

print("\n===== Training Ridge decoder =====")

model = make_pipeline(
    StandardScaler(),
    Ridge(alpha=ridge_alpha)
)

model.fit(X_train, y_train)

print("Model trained successfully")


# ============================================================
# 9. 测试集预测
# ============================================================

print("\n===== Predicting hand velocity =====")

y_pred = model.predict(X_test)

print("y_pred shape:", y_pred.shape)


# ============================================================
# 10. 评估解码性能
# ============================================================

print("\n===== Decoding Performance =====")

r2_x = r2_score(y_test[:, 0], y_pred[:, 0])
r2_y = r2_score(y_test[:, 1], y_pred[:, 1])
r2_overall = r2_score(y_test, y_pred)

mse_x = mean_squared_error(y_test[:, 0], y_pred[:, 0])
mse_y = mean_squared_error(y_test[:, 1], y_pred[:, 1])
mse_overall = mean_squared_error(y_test, y_pred)

corr_x = np.corrcoef(y_test[:, 0], y_pred[:, 0])[0, 1]
corr_y = np.corrcoef(y_test[:, 1], y_pred[:, 1])[0, 1]

print("R2 velocity x:", r2_x)
print("R2 velocity y:", r2_y)
print("R2 overall:", r2_overall)

print("MSE velocity x:", mse_x)
print("MSE velocity y:", mse_y)
print("MSE overall:", mse_overall)

print("Correlation velocity x:", corr_x)
print("Correlation velocity y:", corr_y)


# ============================================================
# 11. 保存模型
# ============================================================

print("\n===== Saving model =====")

joblib.dump(model, model_save_path)

print("Model saved to:", model_save_path)


# ============================================================
# 12. 可视化：真实速度 vs 预测速度
# ============================================================

print("\n===== Plotting results =====")

plot_len = min(plot_len, len(y_test))

time = np.arange(plot_len) * bin_size / 1000.0

plt.figure(figsize=(14, 7))

plt.subplot(2, 1, 1)
plt.plot(time, y_test[:plot_len, 0], label="True velocity x", linewidth=1)
plt.plot(time, y_pred[:plot_len, 0], label="Predicted velocity x", linewidth=1)
plt.ylabel("Velocity X")
plt.title("Basic Ridge Decoder: Hand Velocity X")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, y_test[:plot_len, 1], label="True velocity y", linewidth=1)
plt.plot(time, y_pred[:plot_len, 1], label="Predicted velocity y", linewidth=1)
plt.xlabel("Time (s)")
plt.ylabel("Velocity Y")
plt.title("Basic Ridge Decoder: Hand Velocity Y")
plt.legend()

plt.tight_layout()
plt.savefig(result_fig_path, dpi=300)
plt.show()

print("Figure saved to:", result_fig_path)


# ============================================================
# 13. 可视化：真实速度二维散点 vs 预测速度二维散点
# ============================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test[:plot_len, 0], y_pred[:plot_len, 0], s=5, alpha=0.5)
plt.xlabel("True velocity x")
plt.ylabel("Predicted velocity x")
plt.title("Velocity X: True vs Predicted")

plt.subplot(1, 2, 2)
plt.scatter(y_test[:plot_len, 1], y_pred[:plot_len, 1], s=5, alpha=0.5)
plt.xlabel("True velocity y")
plt.ylabel("Predicted velocity y")
plt.title("Velocity Y: True vs Predicted")

plt.tight_layout()
plt.savefig("basic_ridge_scatter.png", dpi=300)
plt.show()

print("Scatter figure saved to: basic_ridge_scatter.png")


# ============================================================
# 14. 通过速度积分得到二维轨迹
# ============================================================

print("\n===== Reconstructing position from velocity =====")

# 速度单位具体取决于数据，本处只是做相对轨迹可视化
dt = bin_size / 1000.0

true_pos = np.cumsum(y_test[:plot_len, :] * dt, axis=0)
pred_pos = np.cumsum(y_pred[:plot_len, :] * dt, axis=0)

plt.figure(figsize=(7, 7))

plt.plot(true_pos[:, 0], true_pos[:, 1], label="True trajectory", linewidth=2)
plt.plot(pred_pos[:, 0], pred_pos[:, 1], label="Decoded trajectory", linewidth=2)

plt.xlabel("X position")
plt.ylabel("Y position")
plt.title("Reconstructed Trajectory from Decoded Velocity")
plt.legend()
plt.axis("equal")
plt.tight_layout()
plt.savefig("basic_ridge_trajectory.png", dpi=300)
plt.show()

print("Trajectory figure saved to: basic_ridge_trajectory.png")


print("\n===== Done =====")