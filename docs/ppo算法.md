\documentclass{ctexart}
\usepackage{amsmath,amssymb}
\usepackage{geometry}
\geometry{a4paper, margin=1in}

\begin{document}

\section*{PPO 算法训练流程详解}

本部分详细拆解 PPO (Proximal Policy Optimization) 在大模型训练中的核心执行逻辑。

\subsection*{第一阶段：数据生成 (Rollout / Experience Collection)}

此阶段为 Actor 模型 (Qwen2.5-3B) 与环境的交互过程：

\begin{enumerate}
    \item \textbf{Prompt 输入}：从训练集中提取 Prompt。
    \item \textbf{Actor 采样 (Sampling)}：Actor 根据当前权重 $\theta_{old}$，对输入的 Prompt 进行 autoregressive (自回归) 采样，生成对应的回复。
    
    \begin{itemize}
        \item \textbf{注意}：除生成回复外，系统还会记录模型对每个 token 的对数概率 (Log Probabilities)，这是后续计算 \texttt{ratio} 的关键。
    \end{itemize}
    
    \item \textbf{Reward 打分 (Reward Modeling)}：
    
    \begin{itemize}
        \item 将生成的回复送入 Reward 模型 (如 DeepSeek-v4-flash)。
        \item Reward 模型输出标量分数 $r_i$。
        \item \textbf{关键处理}：不仅使用 $r_i$，通常还会结合 Reference 模型 (冻结的初始模型) 计算 KL 散度惩罚。
        \item \textbf{最终奖励公式}：$R_i = r_i - \beta \cdot \text{KL}(\text{Actor} \parallel \text{Ref})$
        \item \textbf{作用}：防止 Actor 为骗取高分而产生乱码或偏离原始语言风格。
    \end{itemize}
\end{enumerate}

\subsection*{第二阶段：策略优化 (Policy Optimization / Gradient Update)}

本阶段利用样本的 $R_i$ 进行模型参数 $\theta$ 的更新：

\subsubsection*{1. 优势估计 (Advantage Estimation)}

Critic 模型 (Qwen2.5-3B，参数不同) 在此发挥作用：

\begin{itemize}
    \item \textbf{价值评估}：Critic 对 80 个样本的当前状态评估一个价值 $V_i$。
    \item \textbf{优势函数}：$A_i = R_i - V_i$。
    \item \textbf{解释}：若 $A_i > 0$，说明生成的回复优于 Critic 的预期，Actor 应增加生成该回复的概率。
\end{itemize}

\subsubsection*{2. 计算 PPO Loss (目标函数)}

不再直接使用 $A_i$ 更新，而是通过对比新旧策略的 \texttt{ratio}：

\[
\text{ratio} = \frac{\pi_{\theta}(\text{response}|\text{prompt})}{\pi_{\theta_{old}}(\text{response}|\text{prompt})}
\]

该 \texttt{ratio} 反映了当前模型对于“高分回复”的倾向程度。

\textbf{目标函数公式}：

\[
L(\theta) = \mathbb{E}\left[\min\left(\text{ratio} \cdot A_i, \text{clip}(\text{ratio}, 1 - \epsilon, 1 + \epsilon) \cdot A_i\right)\right]
\]

\begin{itemize}
    \item \textbf{Clip 的精髓}：强制 \texttt{ratio} 限制在 $[1-\epsilon, 1+\epsilon]$ 之间 (通常为 0.8 到 1.2)。
    \item \textbf{物理意义}：若新模型在某样本上改进过激，\texttt{ratio} 超过范围，clip 操作将进行截断。这保证了模型更新的平稳性，避免因个别高分样本导致灾难性偏移。
\end{itemize}

\subsubsection*{3. 梯度反向传播 (Backward Pass)}

\begin{itemize}
    \item 计算目标函数 $L(\theta)$ 对模型参数 $\theta$ 的梯度 $\nabla_{\theta}L(\theta)$。
    \item 使用 Adam 等优化器执行梯度下降。
    \item \textbf{更新完成}：Actor 模型演进为更优版本 $\theta_{new}$。
\end{itemize}

\subsection*{整个闭环示例 (以 prompt\_idx = 0 为例)}

假设 prompt\_idx = 0 的 Reward 分数很高，Actor 在该样本上表现极佳：

\begin{enumerate}
    \item \textbf{Actor}：生成该回复的概率增大。
    \item \textbf{Critic}：感知到该输入能带来高分，下次预测 $V_i$ 时会提高该状态的评分。
    \item \textbf{Reference}：计算与原始 Qwen2.5-3B 的 KL 散度。若 Actor 走得太远 (变得不认识原始语言)，KL 惩罚项将变大并拉回 Actor。
    \item \textbf{反向传播}：模型权重朝着“生成类似 prompt\_idx=0 高分回复的概率方向”移动了一小步。
\end{enumerate}

\end{document}
