# 数据解码处理
## 1. 基础运动解码模型
### 模型已经能够从 motor cortex spikes 中解码出一部分手部速度信息
### 完整流程
#### NWB 数据读取
#### → 提取 spikes 和 hand_vel
#### → 1 ms 数据 binning 到 20 ms
#### → 构造过去 100 ms spike history
#### → 训练 Ridge decoder
#### → 测试集预测 hand velocity
#### → 计算 R² / MSE / correlation
#### → 保存模型和图片
### 结果示例
#### Basic_ridge_decoding_result
<img width="4200" height="2100" alt="basic_ridge_decoding_result" src="https://github.com/user-attachments/assets/31e89044-2a32-48f2-bd06-aba39be64010" />

#### Basic_ridge_scatter
<img width="3600" height="1500" alt="basic_ridge_scatter" src="https://github.com/user-attachments/assets/f1f9d6e5-2352-468e-89e1-b02e1c07f67a" />

#### Basic_ridge_trajectory
<img width="2100" height="2100" alt="basic_ridge_trajectory" src="https://github.com/user-attachments/assets/f1810c78-3f0b-4b9c-be48-79dfce9d93c1" />

## 2. 做 Ridge 参数扫描
### 现在的参数
#### bin_size = 20 (20 ms bin)
#### history_bins = 5 (过去 100 ms 神经活动)
#### ridge_alpha = 100.0 (Ridge alpha = 100)

### Ridge baseline
| Model | Split | Bin size | Spike history | Alpha | R2 overall | Corr x | Corr y |
|--------|--------|--------|--------|--------|--------|--------|--------|
| Basic Ridge | chronological 80/20 | 20 ms | 100 ms | 100 | 0.331 | 0.603 | 0.547 |
| Tuned Ridge | chronological 80/20 | 50 ms | 500 ms | 1000 | 0.553 | 0.751 | 0.737 |
