<img width="3520" height="1100" alt="ai_score_iteration_mean" src="https://github.com/user-attachments/assets/1ee62872-34eb-4df8-9a4a-388b1d37a504" />


<br>
### 1.扩散模型在第五轮时学习了高分的数据，导致了过拟合，然后此后生成的都与实际不符<br>

* ### 偏差产生：如果 $\tau_4$ 中包含了一些虽然得分高、但逻辑上具有局限性或特征单一的路径（即过拟合），扩散模型会误以为这是“通用的高分范式”。<br>

* #### 查看在第 4 轮和第 5 轮中，模型生成的哪些样本获得了“虚高”的奖励（1.提取第4轮和第5轮ppo_data中的response和对应的rewards。2.比对分析找出在第 5 轮中，哪些样本的 rewards 很高，但orm_score（在 metadata 或 api_scores 中）却很低。）<br>
  <img width="880" height="345" alt="image" src="https://github.com/user-attachments/assets/28296aed-a056-4e43-9a9e-f041f8c57929" />
分析：扩散模型学会了只要是含xml结构的都会给高分，但是deepseek打的客观分只看有没有禁左，看完成度，可以看到deepseek直接给这样的回答打0分，而模型也学会了输出这样的结构去给prm骗高分，所以后面看到的高分的R其实都是xml结构的高分，扩散模型已经完全学会了这种高分奖励<br>


* #### 扩散模型如果过拟合，它对 $\tau$ 的分布会变得非常狭窄（1.提取 hidden_states（即 $\tau$）的张量。2.可视化分布的去计算每一轮 $\tau$ 的均值和方差。3.聚类分析：使用 PCA将第 4 轮和第 5 轮的 $\tau$ 投影到二维平面。）
--- 隐状态统计分析 ---
第4轮方差(std): 1.521111
第5轮方差(std): 1.519077
<img width="1000" height="600" alt="tau_distribution_pca" src="https://github.com/user-attachments/assets/6f0308f5-dd35-4cd5-ac33-c48e187b5ff9" />
分析：方差几乎没有变化，说明第5轮模型生成的隐状态集合，在特征空间中的覆盖范围与第4轮完全一致，排除了模型只会生成一种回答的可能，也没有丢失原本的推理能力；PCA投影图中第五轮有部分红点出现了，第四轮蓝点没有覆盖的区域，说明在第五轮强行探索了新的、且此前从未出现过的隐状态模式。<br>

* #### 检查是否是在第四轮时出现rho是0分但奖励偏高，且回答是xml格式的部分<br>
<img width="979" height="563" alt="image" src="https://github.com/user-attachments/assets/a6e11378-c3d1-4d38-b439-e1939e8a6c65" />
<img width="614" height="191" alt="image" src="https://github.com/user-attachments/assets/48c75e79-da8c-402b-a1c0-2d0966dbae83" />
分析：<br>
1.ai打分理由：几乎rho为的理由都是重复路网文件和不完整的xml标签<br>
2.38 个回答的高分hidden_states趋于一致，如果这些样本的 prm 高分不是因为 XML 格式，那它们之间就不会表现出这种高度的特征聚集。<br>
3.PCA图表明出现的新探索即是扩散模型学习后诱导的结果<br>

结论：扩散模型已经将“输入含有 <net> 等 XML 格式”这一统计特征，与“高奖励”建立起了错误的因果关联<br>

### 修改方案
* 方案A：在训练扩散模型前，设定阈值进行过滤，设计动态自适应阈值，随着迭代次数的增加，适当调高阈值。用高分样本去训练扩散模型并且多采样几次<br>
* 方案B：引入step_mask强相关性奖励，代码里有step_mask逻辑，确保prm奖励必须高度依赖于step_mask 的命中情况，如果模型只输出了 XML 而没有命中任何“禁止左转”的关键操作步，即使 XML 格式再标准，奖励也应趋近于 0。<br>
* 方案C：加入惩罚机制，对复读行为增加负反馈<br>
* 方案D：强制采样负样本,将这些偏高分的奖励标注其rho是0，去训练扩散模型<br>
* 实现优先级：A>B>C>D


