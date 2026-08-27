# Neurodata Without Borders (NWB) 数据格式
## 1. 简介
### 它是一种专门用于神经科学实验数据的标准格式，底层通常是 HDF5
### 可以同时存储：
#### 神经元放电数据，例如 spike counts
#### 行为数据，例如手的位置、速度
#### 实验事件时间，例如 target onset、go cue、movement onset
#### trial 信息，例如 trial_id、target 位置、是否成功
#### 元数据，例如动物、脑区、采样率、实验任务等
## 2. 处理
### nlb_tools 把 NWB 文件转换成了 pandas DataFrame 格式，所以现在看到的是表格
#### 2.1 输出 dataset.data.shape 如下:
##### [5 rows x 190 columns] (6952301, 190)
##### 说明:有 6,952,301 行, 有 190 列, 每一行代表一个时间点, 连续时间数据
#### 2.2 dataset.data 的列结构：MultiIndex
#### 输出的列名
##### Index(['cursor_pos', 'eye_pos', 'hand_pos', 'hand_vel', 'heldout_spikes','spikes'], dtype='object', name='signal_type')
#### 2.3 各个 signal type 的含义
#### 2.3.1 cursor_pos (对应 cursor position)
##### 屏幕上光标的二维位置 (cursor_pos x & cursor_pos y)
#### 2.3.2 eye_pos ( 对应 eye position)
##### 眼睛位置, 这个信号通常用于检查动物是否注视，或者作为行为控制变量
#### 2.3.3 hand_pos
#### 2.3.4 hand_vel
#### 2.3.5 spikes
#### 2.3.6 heldout_spikes


