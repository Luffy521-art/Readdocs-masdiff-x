<img width="3520" height="1100" alt="hidden_state_trend" src="https://github.com/user-attachments/assets/6cca9e26-8666-4177-a064-d96d38ec6e39" />
Hidden State Trend
纵轴是 hidden score，横轴是 sample index，右侧还叠了 valid count。这里的曲线不是时间序列，而是每一轮中样本统计后的均值、中位数、最小值、最大值和四分位带。看法是看不同轮次的分布是否整体抬升、收缩或变散。它表示 hidden state 的整体幅度变化。效果是帮助你判断训练后表征是否更集中、更分散，或者是否存在离群点。

<img width="3520" height="1100" alt="hidden_state_heatmap" src="https://github.com/user-attachments/assets/ad1b3b5c-9654-4e91-b674-4ec52ee66344" />
Hidden State Heatmap
纵轴是 iteration，横轴是 sample index，颜色表示每个样本的 hidden state 强度。看法是：同一行看这一轮里不同样本的分布差异，不同行看不同轮次是否整体变亮、变暗或出现局部热点。它表示模型内部表征在样本上的空间分布，适合发现某些样本是否持续激活更强，或者某一轮是否出现异常峰值。效果是直观看出“哪些样本更敏感，哪些轮次更不稳定”。

<img width="3520" height="1100" alt="hidden_state_delta" src="https://github.com/user-attachments/assets/db5d750d-664b-4769-bd50-ec788e209264" />
Hidden State Mean Delta
纵轴是 delta，横轴是 iter transition，表示相邻两轮之间 hidden score 的平均变化和中位数变化。看法是：大于 0 说明后面一轮比前一轮整体变大，小于 0 说明整体变小。它表示 hidden state 在迭代中的变化方向和幅度。效果是帮助你判断模型表征是在持续增强、持续衰减，还是只是在某几轮出现波动。

<img width="3520" height="1100" alt="prm_trend" src="https://github.com/user-attachments/assets/29922b39-7253-4d06-b0e1-484fce695511" />
PRM Trend
纵轴是 prm score，横轴是 sample index，右侧是 valid count。它展示每轮样本的统计趋势，包括均值、中位数、四分位和极值。看法是看整轮 PRM 是否上移、下移，或者极值是否拉开。它表示奖励水平在样本维度上的总体形态。效果是辅助判断训练过程是否真的把高奖励样本推高，或者只是少数样本抬升。

<img width="3520" height="1100" alt="prm_heatmap" src="https://github.com/user-attachments/assets/d3aa4bc9-a5f0-40f4-b0a0-6e2c3431d3a4" />
PRM Heatmap
纵轴是 iteration，横轴是 sample index，颜色表示每个样本的 PRM 分数。看法和 hidden heatmap 类似，但关注的是奖励模型或过程奖励信号，而不是表征本身。它表示每轮各样本的奖励强弱分布，能看出哪些样本更容易拿到高分、哪些轮次整体奖励更高。效果是判断 PRM 是否稳定，以及训练是否让奖励分布更集中或更分化。

<img width="3520" height="1100" alt="prm_delta" src="https://github.com/user-attachments/assets/85f233ca-ef54-43fa-a6c8-dc2a4af06ca5" />
PRM Mean Delta
纵轴是 delta，横轴是 iter transition，表示相邻轮 PRM 平均变化。看法是正值代表下一轮 PRM 更高，负值代表下一轮更低。它表示奖励信号在迭代中的推进方向。效果是快速看出哪一轮发生了明显奖励提升或退化，比直接看趋势图更容易定位“哪一步开始变好/变坏”。

<img width="3520" height="1100" alt="ai_score_trend" src="https://github.com/user-attachments/assets/511f3d27-b802-4ff8-bfa5-141e92718497" />
AI Score Trend
纵轴是 ai score，横轴是 sample index，右侧是 valid count。这里展示的是每轮样本的统计分布，不是单个样本的时间变化。看法是看均值、中位数、四分位和极值，判断打分是否集中、是否有长尾、是否轮次间有抬升。它表示 AI 评分的总体水平。效果是帮你判断模型输出质量是否在迭代中稳定改善。

<img width="3520" height="1100" alt="ai_score_heatmap" src="https://github.com/user-attachments/assets/b5bc04b7-b3e2-4e87-bcad-5a2a283f4adf" />
AI Score Heatmap
纵轴是 iteration，横轴是 sample index，颜色表示每个样本的 AI 打分。看法是同一轮里看样本之间分数分布，不同轮里看整体是否偏高或偏低。它表示外部评审/打分器对样本质量的判断，最适合看哪些样本一直得低分，哪些轮次整体更优。效果是把“质量差异”直接可视化，比只看均值更直观。

<img width="3520" height="1100" alt="ai_score_delta" src="https://github.com/user-attachments/assets/10b6e504-a612-46cf-a701-2ce617b460b6" />
AI Score Mean Delta
纵轴是 delta，横轴是 iter transition，表示相邻轮 AI score 的平均变化。看法是正值表示下一轮整体评分更高，负值表示更低。它表示样本质量在迭代中的提升或退步。效果是快速定位训练是否真的带来了质量变化，特别适合和 hidden state delta、PRM delta 一起对照看。
这张图展示的是各轮次的 AI score 分布，横轴是 iteration，纵轴是 ai score。可以看出第 4 轮分数最高，第 2 轮最低，第 5、6 轮回升但仍不如第 4 轮，说明模型输出质量在轮次间有波动而不是持续单调提升。箱体几乎压成线，说明每轮样本分数比较集中，分布离散度不大，适合用来比较不同轮次的整体质量变化。

<img width="3520" height="1100" alt="ai_score_iteration_mean" src="https://github.com/user-attachments/assets/7c1fcf07-2e39-415c-8e9b-1263dc6fe739" />
AI Score Mean by Iteration
纵轴是 ai score，横轴是 iteration，画的是每一轮的均值和中位数。看法是看曲线是否逐轮上升、下降或波动。它表示整体质量随训练轮次的变化轨迹。效果是最适合做总览图，直接回答“模型平均质量有没有提高”，比逐样本图更容易读。
