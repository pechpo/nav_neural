# PointNav 导航模型中的自我中心空间表征与障碍物记忆机制分析

**课程名称**：系统与计算神经科学
**小组成员与分工**：

* **彭程（组长）**：项目统筹；代码编写；做实验；报告撰写与可视化整理。
* **[组员1姓名]**：
* **[组员2姓名]**：

## 1. 动机与背景

2018 年发表在 *Nature* 的工作 *Vector-based navigation using grid-like representations in artificial agents* 展示了一个很打动人的现象：人工智能体在学习导航/路径积分时，内部会自发涌现出类似生物系统的“网格样”表征。这件事给我们的启发是——导航网络不一定只是“黑盒策略”，它里面可能真的存在可解释的空间信息组织方式。

受此启发，我们试图在经典的 PointNav 模型中，讨论 agent 如何构建**自我中心（egocentric）**的空间感知？尤其是：
- 视野内（眼睛看得到）的障碍物信息，当然可以来自 CNN；
- 那些刚刚移出视野、或者在侧后方盲区的障碍物信息，网络是否会在 RNN 的隐状态中保留某种短时记忆（有点像“客体恒常性 / object permanence”）？

我们选择了一个相对朴素但易解释的切入点：把“障碍物”拆成 **12 个方向上的最近障碍物距离**，然后用线性探针去问：这些距离在模型的哪些表示里是线性可读出的？

---

## 2. 实验设置与数据采集

### 2.1 仿真环境与轨迹生成（Random Frontier）

实验平台：Habitat；场景：HM3D；episode：ObjectNav（我们借用它的多样室内场景与初始位姿）。

<table>
  <tr>
    <td><img src="res/1.png" width="300"></td>
    <td><img src="res/2.png" width="300"></td>
  </tr>
  <tr>
    <td colspan="2" align="center">
      <b>模拟器运行示例</b>
    </td>
  </tr>
</table>

为了避免标准 PointNav “直奔目标”带来的强目标偏置，我们实现/使用了 `RandomFrontierPlanner`（`neural/planner/random_frontier_planner.py`）：
1) 根据当前 `ObstacleMap` 识别 frontier（已探索与未探索区域的边界）；
2) 从可达 frontier 里随机选一个作为临时目标；
3) 让 agent 持续走、持续转向，尽量覆盖多方向、多距离的障碍情形。

这套“随机探索 + 广泛转向”的采样方式对线性探针很友好：数据分布更丰富，不容易只学到“前进方向上那堵墙”。

### 2.2 障碍物真值（12 方向 raycast 距离）

每一步我们用 `ObstacleMap.raycast_obstacle_distances()` 在 agent 周围均匀打 12 条射线（默认 `max_range=5m`），取最近命中的障碍距离，得到 `distances`（shape: `N×12`）。

方向编号规则：
- `dir_0`：正前方
- `dir_1`：右前方
- ...
- `dir_3`：正右方
- `dir_6`：正后方
- `dir_9`：正左方
- `dir_11`：左前方

> 注：方向从 `dir_0` 开始，按顺时针均匀采样（与 `neural/mapping/obstacle_map.py` 中的角度定义保持一致）。

### 2.3 模型架构与“探针点”

我们使用 VLFM 提供的预训练 `PointNavResNetPolicy` checkpoint，把网络内部表示拆成三块记录（共 3072 维）：
1) **Visual Embed（层号 0）**：`activations_visual_embed`，512 维（偏“眼睛看到什么”）
2) **RNN Hidden（层号 1~4）**：`activations_hidden`，4×512=2048 维  
   - 这里的 “4” 来自 LSTM 的状态结构：两层 LSTM ×（h, c）两种状态
3) **RNN Output（层号 5）**：`activations_rnn_output`，512 维（更接近“要做什么动作”）

### 2.4 视野内 / 视野外方向

我们用相机水平视场角（HFOV）在 12 个方向上生成可见性 mask（见 `neural/eval/worker.py`）。在当前这次结果配置（`result/analysis_out_hidden/config.json`）里：
- **视野内（On-screen）**：`dir_0, dir_1, dir_11`
- **视野外（Off-screen）**：其余 9 个方向

---

## 3. 分析方法（两阶段线性探针）

<table>
    <tr>
        <td>
            <img src="res/analysis_out_hidden/heat_dir_1_rank1_rnn_hidden-L1-U24.png" width="260">
        </td>
        <td>
            <img src="res/analysis_out_hidden/heat_dir_3_rank1_rnn_hidden-L3-U20.png" width="260">
        </td>
        <td>
            <img src="res/analysis_out_hidden/heat_dir_8_rank3_rnn_hidden-L3-U239.png" width="260">
        </td>
    </tr>
    <tr>
        <td align="center">对约一点钟方向<br>负反应的神经元</td>
        <td align="center">对约三点钟方向<br>正反应的神经元</td>
        <td align="center">对约八点钟方向<br>正反应的神经元</td>
    </tr>
</table>

数据量大（当前结果的样本数 `N=366,018`，见 `result/analysis_out_hidden/config.json`），如果直接对全部 3072 维做回归，一方面计算量高，另一方面会被大量弱相关维度拖累。我们用“初筛 + 稀疏回归”两步走：

### 3.1 Spearman 相关性筛选（screening）

对每个方向 `dir_k`，计算所有神经元激活与该方向距离的 Spearman 相关系数（`neural/models/spearman.py`），然后：
- 取 `abs(corr)` 排序
- 保留 Top-50
- 且要求 `abs(corr) >= 0.2`

筛选结果保存为 `screened_neurons.json`，里面除了相关值，还会写入 `feature_meta(part/layer/unit)` 方便我们定位“哪个层的哪个单元”。

为什么用 Spearman：距离-激活关系不一定线性（更像单调非线性），Spearman 对这类关系更鲁棒。

### 3.2 Lasso 回归（L1 线性探针）

对每个方向，把筛选出来的特征喂给 Lasso（`neural/models/lasso.py`），用线性模型去拟合该方向的距离，输出 `lasso_metrics.json`：
- `mae`：平均绝对误差（越小越好）
- `n_features`：入模特征数（即筛选后留下的特征数量，上限 50）

> 注：这里的 `n_features` 是“进入 Lasso 的特征数”，不是非零权重数；如果想看真正稀疏后的非零权重数量，可以结合 `lasso_weights.npy` 统计。

### 3.3 Top-3 代表性神经元汇总（跨来源混合）

为了回答“这个方向到底主要靠哪一类表示在编码”，我们允许 top-3 神经元来自 `visual / hidden / output` 任意来源：把三套 `screened_neurons.json` 合并后按 `abs_corr` 全局排序取 top3。  
汇总脚本：`result/analyze.py`，输出 `result/merged_summary.csv`（并按 `mae` 从小到大排序）。

---

## 4. 实验结果与观察

### 4.1 视野内 vs 视野外：准确度、层级与特征数的系统差异

基于 `result/merged_summary.csv` 的汇总统计：
- **MAE**：视野内平均 `0.621`，视野外平均 `0.903`
- **n_features**：视野内平均 `50.0`，视野外平均 `29.78`
- **Top-3 平均层号**：视野内 `1.89`，视野外 `3.44`
- **Top-3 平均相关强度**：视野内 `0.594`，视野外 `0.321`

直观理解：视野内的障碍信息更像“视觉直接可读”，更浅层、更密；视野外的障碍信息更像“靠记忆/整合推断”，更深层、更稀疏，但仍然存在。

### 4.2 视野外也能预测：盲区不是“全忘了”

虽然视野外更难，但并非不可预测。比如：
- 部分视野外方向依然能筛到大量相关神经元（`n_features=50`）并获得中等 MAE；
- 在一些盲区方向的 top-3 中会出现 `output` 来源（例如 `dir_4/dir_8/dir_10`），说明信息可能沿着 RNN 的决策通路进一步“沉淀”。

### 4.3 各方向汇总表（按 MAE 升序）

| dir | top3_sources | top3_abs_corr_mean | top3_layer_mean | mae | n_features |
| --- | --- | --- | --- | --- | --- |
| dir_1 | hidden,hidden,visual | 0.5716 | 1.33 | 0.5744 | 50 |
| dir_11 | hidden,hidden,visual | 0.5666 | 2.00 | 0.6051 | 50 |
| dir_0 | hidden,hidden,hidden | 0.6437 | 2.33 | 0.6843 | 50 |
| dir_2 | hidden,hidden,hidden | 0.4460 | 3.33 | 0.7458 | 50 |
| dir_10 | hidden,hidden,output | 0.4224 | 3.67 | 0.7790 | 50 |
| dir_3 | hidden,hidden,hidden | 0.3347 | 2.67 | 0.8378 | 47 |
| dir_9 | hidden,hidden,hidden | 0.3456 | 3.67 | 0.8680 | 44 |
| dir_4 | hidden,output,hidden | 0.2569 | 3.67 | 0.9008 | 9 |
| dir_8 | hidden,hidden,output | 0.2716 | 3.67 | 0.9312 | 6 |
| dir_7 | hidden,hidden,hidden | 0.2228 | 3.00 | 0.9985 | 8 |
| dir_6 | hidden,hidden,hidden | 0.3836 | 3.67 | 1.0243 | 50 |
| dir_5 | hidden,hidden,hidden | 0.2057 | 3.67 | 1.0413 | 4 |

---

## 5. 讨论：祖母细胞假说不成立，更像群体编码

如果“祖母细胞”假说成立，我们会期待看到：某些方向上存在极少数、甚至单个神经元对该方向障碍距离呈现极强（接近 1）的可读性，并且模型主要依赖这些“明星单元”。但本次结果更像群体编码：
- 单个神经元相关性不“极端”（最高也就 ~0.66 左右），没有“一个神经元就够了”的感觉；
- 预测好的方向往往伴随更多特征通过筛选（视野内 `n_features=50` 很典型），而特征稀疏时性能明显下降（例如 `dir_5` 只有 4 个特征，MAE 最大）；
- top-3 神经元来源跨部件/跨层（hidden/visual/output 都会出现），说明信息表征更像是网络整体动力学与群体活动的结果，而不是某个“单细胞结论”。

换句话说：PointNav 模型里的“避障记忆”，更像是群体细胞一起投票出来的。

---

## 6. 复现方式

请参考 [复现流程](./Reproducible_Pipeline.md) 了解更多。