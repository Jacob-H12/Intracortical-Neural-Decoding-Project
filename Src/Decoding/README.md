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
#### 2.3.2 eye_pos (对应 eye position)
##### 眼睛位置, 这个信号通常用于检查动物是否注视，或者作为行为控制变量
#### 2.3.3 hand_pos (对应 hand position)
##### 输出 hand_pos shape: (6952301, 2), 说明: 6,952,301 个时间点,每个时间点有 x、y 两个坐标, 就是每 1 ms 的手部位置
#### 2.3.4 hand_vel (对应 hand velocity)
##### 输出 hand_vel shape: (6952301, 2), 说明: 每个时间点都有 x 方向速度, 每个时间点都有 y 方向速度
##### 做运动解码时很重要。可以用神经活动预测 hand_vel_x & hand_vel_y
#### 2.3.5 spikes (主要的神经数据)
##### 输出 spikes shape: (6952301, 137), 时间点数：6,952,301, 神经通道数：137. 137 个神经元/神经通道, 每 1 ms 一个 spike count
##### 例如：clock_time: 0 days 00:00:00.001000      2931: 1.0. 这表示在 1 ms 这个时间 bin，编号为 2931 的神经通道发放了一个 spike.
#### 2.3.6 heldout_spikes ( 对应 hand position)
##### 输出 heldout_spikes shape: (6952301, 45), 45 个 held-out 神经通道
##### 在 benchmark 任务中，模型可能只能看到 spikes，然后要预测或者重建 heldout_spikes
#### 2.4 dataset.trial_info：每个 trial 的信息
##### 输出 trial_info shape: (2295, 18), 2295 个 trials, 每个 trial 有 18 个字段
##### 如下 Index(['trial_id', 'start_time', 'end_time', 'trial_type', 'trial_version','maze_id', 'success', 'target_on_time', 'go_cue_time','move_onset_time', 'rt', 'delay', 'num_targets', 'target_pos','num_barriers', 'barrier_pos', 'active_target', 'split'],dtype='object')
#### 2.4.1 trial_id
##### trial 编号
#### 2.4.2 start_time
##### trial 开始的全局时间
#### 2.4.3 end_time
##### trial 结束的全局时间
#### 2.4.4 trial_type
##### 试次类型, 通常代表不同条件，例如不同目标、不同 maze layout、不同障碍物配置
##### 做条件平均 PSTH 的时候，可以按trial_type分组
#### 2.4.5 trial_version
##### trial 版本, 通常是任务设计中的版本标签，可能表示同一种 trial_type 下的不同配置版本
#### 2.4.6 maze_id
##### maze 编号, 就是迷宫配置编号, MC_Maze 是 maze reaching task，所以不同 trial 可能对应不同 maze layout
##### 要按迷宫条件分析神经轨迹，可以按maze_id分组
#### 2.4.7 success
##### 是否成功完成 trial
#### 2.4.8 target_on_time
##### 目标出现时间, 这是全局时间
##### 在 delayed reaching task 中，常见流程是: trial start -> target appears -> delay period -> go cue -> movement onset -> movement end
#### 2.4.9 go_cue_time
##### go cue 出现时间, 这是允许猴子开始运动的信号时间
#### 2.4.10 move_onset_time
##### 运动真正开始的时间, 这是行为学定义的 movement onset, 通常基于手速度阈值检测
#### 2.4.11 rt
##### reaction time, 反应时, 也就是从 go cue 到真正开始运动之间的时间
#### 2.4.12 delay
##### delay period 长度, 也就是看到目标后，到 go cue 出现之前的等待时间
#### 2.4.13 num_targets
##### 目标数量, MC_Maze 任务中可能一个 trial 有多个候选目标, active_target 指示当前真正目标是哪一个
#### 2.4.14 target_pos
##### 目标位置, 可能是二维坐标数组
#### 2.4.15 num_barriers
##### 障碍物数量, MC_Maze 的特色就是 maze reaching, 猴子需要绕开 barrier
#### 2.4.16 barrier_pos
##### 障碍物位置, 每个 barrier 用一组数描述, 可以先理解为障碍物在二维空间里的位置和尺寸
#### 2.4.17 active_target
##### 当前 trial 的真实目标编号, 如果一个 trial 中有多个 target，那么 active_target 告诉你猴子实际要到哪一个 target
#### 2.4.18 split
##### 数据划分, 说明 trial 被分成训练集和验证集
#### 2.5 trial_data 和 dataset.data 的区别
##### 2.5.1 dataset.data 连续全局数据
##### 适合: 查看整个实验连续记录, 做连续解码, 按时间窗口手动截取
##### 2.5.2 trial_data 按 trial 切分后的数据
##### 适合：做 PSTH, 做 trial average, 按 go cue / movement onset 对齐, 做 PCA 神经轨迹, 分条件比较神经活动
