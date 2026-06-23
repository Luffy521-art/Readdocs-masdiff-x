# 基于大模型隐状态与 PRM 的 Rho 预测代理模型

## 1 任务定义
本任务旨在训练一个轻量级的代理模型（Surrogate Model）。模型接收大语言模型生成的隐状态特征（$\tau_{hidden}$）和过程奖励打分（$R_{prm}$），输出一个条件概率分布 $p(\rho | \tau_{hidden}, R_{prm}, M_{action})$。

在测试阶段，模型输出 Student-t 分布的参数 $(\mu, \sigma, \nu)$，并以分布均值作为最终预测得分：
$$\hat{\rho} = \mu$$

## 2 数据集特征
数据集由大模型的前向传播无损截取，核心字段包括：
* **`hidden_state`**：大模型输出的高维语义张量，维度为 $[Batch, Seq\_Len, 2560]$。
* **`prm`**：过程奖励模型（PRM）的密集打分，维度为 $[Batch, Seq\_Len]$。
* **`action_mask`**：有效动作掩码，用于区分有效 Token 与 Padding 填充区域。
* **`rho`**：真实环境评估得到的最终得分，作为目标标签。

## 3 模型设计
采用极简联合架构提取并融合特征：
* **动态对齐**：对 `hidden_state`、`prm` 和 `action_mask` 在序列长度上进行动态截断对齐。
* **PRM 升维投影**：通过 `Linear(1, 64)` -> `LayerNorm` -> `GELU`，将 1 维的 PRM 打分升维至 64 维，拉平特征量级。
* **特征拼接与池化**：将 2560 维隐状态与 64 维 PRM 特征拼接为 2624 维。随后结合 `action_mask` 进行掩码均值池化（Masked Mean Pooling），提取全局定长句向量。
* **预测头**：利用两层 MLP（引入 Dropout）将特征映射为 Student-t 分布的三个标量参数 $(\mu, \sigma, \nu)$。

## 4 损失函数设计
模型采用 Student-t 的负对数似然（NLL）作为主损失函数。相比传统的 MSE，Student-t 具有更厚的尾部，能自适应调整置信度 $\sigma$，有效抵抗极端打分（Outliers）的干扰：
$$\mathcal{L}_{NLL} = -\log p(\rho_{true} | \tau_{hidden}, R_{prm}, M_{action})$$
