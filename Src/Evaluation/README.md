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

