# L15 - RLAIF——PPO / GRPO / CISPO

> **"如果人类标注者太贵？让 AI 自己打分，然后让另一个 AI 学那个标准。"**

---

## 📌 本节目标

1. 理解 RLAIF（Reinforcement Learning from AI Feedback）与 RLHF 的区别
2. 掌握 PPO（Policy Gradient）的核心原理
3. 理解 GRPO（Group Relative Policy Optimization）的创新点
4. 了解 minimind 中 `rollout_engine.py` 的解耦架构

---

## 📚 前置知识

- 完成 L14（理解 RLHF 和 DPO）
- 了解强化学习中的"策略梯度"概念

---

## 正文

### 1. 为什么需要 RLAIF

RLHF 的瓶颈很明显——**人类标注太贵**。

- OpenAI 的 RLHF 标注成本：每句话 0.5-1 美元，标注一个 70B 模型需要百万级别的标注数据 → **百万美元成本**

DeepSeek 在 2024 年的论文中提出了一个激进的方法：**用模型自己来标注！**

```
R1 模型（大）
  ↓（生成 2 个回答）
  ├─ 回答 A
  └─ 回答 B
        ↓
      [问自己哪个更好？]
        ↓
    得到偏好标签
        ↓
用 DPO / PPO 训练 R1-Zero（小）
```

**RLAIF vs RLHF：**

| 特性 | RLHF | RLAIF |
|------|------|-------|
| 反馈来源 | 人类标注员 | 大模型自己 |
| 成本 | 超高（$100k-1M） | 低（GPU 时间） |
| 标注一致性 | 低（人类分歧） | 极高（模型确定性） |
| 标注速度 | 慢（需要招人） | 快（推理瞬间完成） |

### 2. PPO 详解：策略梯度的标准做法

**什么是策略梯度（Policy Gradient）？**

强化学习的目标是最大化期望回报：

$$J(\theta) = \mathbb{E}_{y \sim \pi_\theta(y|x)}[\text{Reward}(x, y)]$$

其中 $\pi_\theta$ 是策略（语言模型），$\text{Reward}$ 是奖励函数（RM 或自己的评分）。

策略梯度的技巧是用重要性采样（Importance Sampling）来估计这个梯度：

$$\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_{\text{old}}} \left[ \frac{\pi_\theta(y|x)}{\pi_{\text{old}}(y|x)} \cdot \nabla_\theta \log \pi_\theta(y|x) \cdot A(y) \right]$$

其中 $A(y)$ 是"优势函数"（Advantage），代表这个行动（生成的回答）相对于平均有多好。

**PPO 的关键创新：Clipping**

原版策略梯度容易不稳定。PPO 加了一个"信任区域"约束：

$$L_{\text{PPO}} = \mathbb{E} \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(y_t|x_t)}{\pi_{\text{old}}(y_t|x_t)}$ 是概率比
- $\epsilon$ 是信任区域宽度（通常 0.2），防止参数更新太大

### 3. GRPO 与 CISPO：minimind 的创新

minimind 在 DeepSeek-R1 的启发下，加入了两个优化：

#### GRPO（Group Relative Policy Optimization）

标准的优势函数 $A(y)$ 需要基线函数的估计，容易偏差。GRPO 的想法：**相对比较而不是绝对评分**

```python
def compute_advantage_grpo(rewards):
    """
    给定一组回答的奖励，计算相对优势
    
    假设生成了 K 个回答，奖励为 [r1, r2, ..., rK]
    """
    # 分组：前一半为 "好"，后一半为 "坏"
    group_size = len(rewards) // 2
    good_rewards = rewards[:group_size].mean()
    bad_rewards = rewards[group_size:].mean()
    
    # 优势 = 好组的平均奖励 - 坏组的平均奖励
    # 这样自动做了归一化，更稳定
    advantage = good_rewards - bad_rewards
    
    return advantage
```

优势：
- 无需基线函数，减少方差
- 自动归一化，鲁棒性强
- 与 LLM 的"相对概念"更匹配

#### CISPO（Contrastive Implicit Preference Optimization）

When there's are multiple generations, sometimes simpler approaches compare them directly:

```python
def cispo_loss(model_logits, ref_logits, winners, losers, beta=0.5):
    """
    直接比较 "赢家" 和 "输家" 两个回答的模型预测
    """
    # 获取每个回答的对数概率
    winner_log_probs = compute_log_prob(model_logits, winners)
    loser_log_probs = compute_log_prob(model_logits, losers)
    
    ref_winner_log_probs = compute_log_prob(ref_logits, winners)
    ref_loser_log_probs = compute_log_prob(ref_logits, losers)
    
    # 对比损失：让赢家和输家的差距尽可能大
    model_diff = winner_log_probs - loser_log_probs
    ref_diff = ref_winner_log_probs - ref_loser_log_probs
    
    # 用 beta 加权两个模型的差距
    loss = -torch.log(torch.sigmoid(beta * (model_diff - ref_diff)))
    
    return loss
```

### 4. minimind 的 rollout_engine.py 架构

minimind 用一个解耦的"rollout engine"处理强化学习的"生成-标注-训练"循环：

```python
class RolloutEngine:
    """处理强化学习中的迭代：生成 → 评分 → 更新"""
    
    def __init__(self, model, ref_model, reward_model, tokenizer):
        self.model = model              # 要训练的模型
        self.ref_model = ref_model      # 参考模型（冻结）
        self.reward_model = reward_model # 奖励模型（或评分模型）
        self.tokenizer = tokenizer
    
    def rollout(self, prompt, num_samples=4):
        """
        从 prompt 生成 num_samples 个不同的回答
        """
        generations = []
        for _ in range(num_samples):
            # 生成（采用采样而非贪心）
            output_ids = self.model.generate(
                self.tokenizer.encode(prompt),
                max_new_tokens=128,
                temperature=0.7  # 多样性
            )
            response = self.tokenizer.decode(output_ids)
            generations.append(response)
        
        return generations
    
    def score(self, prompt, responses):
        """
        用奖励模型给每个回答打分
        """
        scores = []
        for response in responses:
            # 拼接 prompt + response
            full_text = prompt + response
            reward = self.reward_model.score(full_text)
            scores.append(reward)
        
        return torch.tensor(scores)
    
    def compute_advantages_grpo(self, scores):
        """
        GRPO：按得分排序，分组计算相对优势
        """
        sorted_indices = torch.argsort(scores, descending=True)
        group_size = len(scores) // 2
        
        good_indices = sorted_indices[:group_size]
        bad_indices = sorted_indices[group_size:]
        
        advantages = torch.zeros_like(scores)
        advantages[good_indices] = 1.0
        advantages[bad_indices] = -1.0
        
        return advantages.unsqueeze(-1)  # 广播维度
    
    def step(self, prompts, optimizer):
        """
        一个完整的强化学习步：生成 → 评分 → 反向传播
        """
        total_loss = 0
        
        for prompt in prompts:
            # 生成
            generations = self.rollout(prompt, num_samples=4)
            
            # 评分
            scores = self.score(prompt, generations)
            
            # 计算优势
            advantages = self.compute_advantages_grpo(scores)
            
            # 计算每个生成的对数概率
            log_probs = []
            for response in generations:
                full_text = prompt + response
                tokens = self.tokenizer.encode(full_text)
                
                outputs = self.model(torch.tensor(tokens).unsqueeze(0))
                logits = outputs.logits[:-1, :]
                labels = tokens[1:]
                
                log_prob = torch.nn.functional.log_softmax(logits, dim=-1)
                log_prob = log_prob[range(len(labels)), labels].sum()
                log_probs.append(log_prob)
            
            log_probs = torch.stack(log_probs)
            
            # PPO loss / GRPO loss
            loss = -(log_probs * advantages).mean()
            total_loss += loss
            loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        return total_loss.item()
```

**架构的优势：**

1. **解耦**：生成、评分、训练完全独立，各自优化
2. **灵活**：RM 可以换成 score 函数、甚至用 API（Llama-2 作评判）
3. **可扩展**：可以轻松加入并行、分布式、缓存等优化
4. **可调试**：每个函数独立可测试

---

## 💡 本节小结

| 概念 | 一句话总结 |
|------|-----------|
| RLAIF | 用大模型自己做反馈，替代昂贵的人类标注 |
| 策略梯度 | 沿着奖励梯度上升调整策略的对数概率 |
| PPO | 在策略梯度的基础上加信任区域（clipping），保证稳定 |
| GRPO | 相对比较优势（前半 > 后半），比绝对评分更稳定 |
| CISPO | 直接对比赢家/输家，隐式学习偏好 |
| Rollout Engine | 解耦的强化学习框架：生成 → 评分 → 优化 |

---

## 🧪 习题集 15

**题目 1（RLAIF vs RLHF）：** 用 RLAIF 时，如何保证奖励模型（或评分模型）自己的标准"合理"？会不会出现"模型自我强化的陷阱"？

**题目 2（PPO Clipping）：** PPO 的 clipping 作用是什么？如果 $\epsilon$ 太大会怎样，$\epsilon$ 太小呢？

**题目 3（优势函数）：** GRPO 中，是否可能所有回答都的优势都是正的？或都是负的？为什么？

**题目 4（代码追踪）：** 在 `RolloutEngine.step()` 中，为什么要对所有 generation 分别计算 log 概率，而不是一次并行计算？

**题目 5（设计题）：** 如果没有显式的奖励模型，只有 "好"/"坏" 的二分类标签，能用 GRPO 吗？怎么修改算法？

---

> 下一课我们将学习 **L16 - 模型蒸馏——知识传承**，用大模型教小模型。
