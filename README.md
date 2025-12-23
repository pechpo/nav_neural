# PointNav 导航模型中的自我中心空间表征与障碍物记忆机制分析

**课程名称**：系统与计算神经科学
**小组成员与分工**：

* **彭程（组长）**：项目统筹；代码编写；做实验；报告撰写与可视化整理。
* **[组员1姓名与分工]**：钟逸，负责理论架构设计；代码逻辑审计；学术综述与文献规范
* **[组员2姓名与分工]**：
* **[组员3姓名与分工]**：

## 1. 动机与背景

### 1.1 研究动机
在系统与神经计算科学的研究范式中，空间导航不仅是一项复杂的行为任务，更是研究大脑如何编码外部世界的窗口。其中，大脑如何通过神经元放电从而构建出精确的空间度量，一直是该任务探索的核心问题。作为利用人工神经网络还原并模拟这一生物特性的关键切入点，**本研究的理论动机源于 Banino 等人在 2018 年发表于《Nature》的研究《Vector-based navigation using grid-like representations in artificial agents》[1]**。该研究通过构建基于长短期记忆网络（LSTM）的循环架构，模拟了生物脑处理运动学信号的过程；同时实验结果表明在执行路径积分任务时，模型内部会自发涌现出具有六边形周期性特征的类网格单元 ，而这一特性恰恰与哺乳动物内嗅皮层中支持空间导航的“网格细胞”高度相似。这一结论不仅揭示了网格细胞在空间导航中的计算逻辑，也证明了深度强化学习模型能够作为模拟空间认知的有效计算工具 。

本研究在此基础上，选取了基于**VLFM（Visual Language Frontier Maps）**预训练背景的**PointNav**模型作为研究对象 [2]。虽然 VLFM 的设计初衷是处理复杂的语义对象导航，但本研究将其应用场景收敛至点对点导航任务。这一选择旨在排除高层语义的干扰，从而能够更纯粹地观察模型在处理环境几何拓扑与障碍物规避时的表征逻辑。利用这一具备强大感知能力的模型，本研究试图探讨人工神经元如何在PointNav任务约束下演化出具有特定功能指向性的计算单元。

### 1.2 相关工作
为了深入挖掘模型内部的计算逻辑，本研究在理论框架上融合了多项跨学科的前沿成果。关于认知地图的构建逻辑，本研究参考了 **Whittington 等人 (2022)在《Nature Neuroscience》** 上的综述 [3]，该文献系统阐述了哺乳动物大脑皮层中的海马体-内嗅皮层系统如何利用结构化表征将空间经验转化为可泛化的规律，为理解模型在不同环境下保持导航一致性提供了坚实的理论基础。在解析特定神经元的功能属性时，本研究借鉴了 **Zhou 等人 (2018)《PNAS》** 上提出的网络单元功能解析思想 [4]，即神经网络内部的单个单元往往会演化为特定视觉概念的检测器。

这一机制的普适性在 **Sorscher 等人 (2023)** 关于网格细胞起源的统一理论中得到了进一步验证：该研究指出，生物化表征的涌现并非偶然，而是特定网络结构在优化空间编码效率时的必然选择 [5]。这一理论为本研究在 VLFM 模型中搜寻避障特化单元提供了合法性依据，也就是说即使任务层级不同，底层的空间感知约束仍会驱动功能性神经元的产生。最后，在工程实现层面，**Tai 等人 (2017)** 证明了利用传感器信息进行端到端导航的有效性 [6]，而 **Fan 等人 (2020)** 关于分散式避障的研究则启发了本研究对模型在动态环境下决策一致性的分析逻辑 [7]。上述这些工作共同构成了本研究从理论基础到项目实践的完整科研链路。

### 1.3 核心目标
本研究的核心目标并不是简单地追求导航任务的成功率，而是利用统计学手段挖掘模型内部针对导航任务的避障机制。实验效仿神经科学中对特定功能神经元的搜寻逻辑，在模型运行过程中同步观测并采集隐层神经元的激活状态，同时在实验阶段引入了斯皮尔曼相关性分析与线性回归模型对神经元活动进行机制解析。

更进一步说，实验过程旨在通过数据采样量化隐层神经元的激活状态与环境障碍物空间分布（如方位角、距离）之间的耦合程度并将其可视化，精准识别出在避障决策中起主导作用的神经元。在这一工作中，我们尝试将模型从感知到行动的复杂过程拆解开来，分析其内部处理机制，尤其关注模型内部是否形成了对空间障碍物等特定信息敏感的神经元。这不仅是对模型认知机制的一种追溯，也为我们理解更大规模、更复杂模型的内部工作原理，提供了一种具体的实证分析思路。


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

## 参考文献 (References)
* [1] Banino A, Barry C, Uria B, et al. Vector-based navigation using grid-like representations in artificial agents[J]. Nature, 2018, 557(7705): 429-433.
* [2] Yokoyama N, Ha S, Batra D, et al. Vlfm: Vision-language frontier maps for zero-shot semantic navigation[C]//2024 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2024: 42-48.
* [3] Whittington J C R, McCaffary D, Bakermans J J W, et al. How to build a cognitive map[J]. Nature neuroscience, 2022, 25(10): 1257-1272.
* [4] Bau D, Zhou B, Khosla A, et al. Network dissection: Quantifying interpretability of deep visual representations[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017: 6541-6549.
* [5] Sorscher B, Mel G, Ganguli S, et al. A unified theory for the origin of grid cells through the lens of pattern formation[J]. Advances in neural information processing systems, 2019, 32.
* [6] Tai L, Paolo G, Liu M. Virtual-to-real deep reinforcement learning: Continuous control of mobile robots for mapless navigation[C]//2017 IEEE/RSJ international conference on intelligent robots and systems (IROS). IEEE, 2017: 31-36.
* [7] Fan T, Long P, Liu W, et al. Fully distributed multi-robot collision avoidance via deep reinforcement learning for safe and efficient navigation in complex scenarios[J]. arXiv preprint arXiv:1808.03841, 2018.
