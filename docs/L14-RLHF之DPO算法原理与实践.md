# L14 - RLHF——DPO 算法原理与实践

> **"与其挨个教会妈妈哪个菜好吃，不如让她尝两道菜然后自己选。"**

---

## 📌 本节目标

1. 理解为什么需要 RLHF：SFT 后模型仍不完美
2. 掌握 RLHF 三阶段架构（SFT → RM → PPO）
3. 理解 DPO（Direct Preference Optimization）如何绕过奖励模型
4. 能实现一个简化的 DPO 训练循环

---

## 📚 前置知识

- 完成 L12（SFT）、L13（LoRA）
- 了解强化学习的基本概念（作用、策略、奖励）

---

## 正文

### 1. SFT 的局限与 RLHF 的动机

完成 SFT 后的模型已经相当不错了，但仍有一些问题：

**问题 1：分布外生成**

SFT 数据中的所有回答都是"标准答案"，模型学不到：

```
Q: 如何制造炸弹？
标准答案（SFT 数据中没有）：我不能帮助这个
实际生成：炸弹是通过…（乱编的危险内容）
```

**问题 2：长序列的优化信号缺失**

SFT 的损失是 token-level 的，模型不知道一个完整答案是"好"还是"坏"：

```
回答 A：大模型是深度学习中一种参数量很大的模型…（冗长但准确）
回答 B：大模型（LLM）是用大数据训练的深度神经网络。（简洁准确）

SFT 无法评判哪个更优，只能说"都对"
RLHF 让人标注偏好（一般选 B），训练模型学会简洁
```

**RLHF 三阶段**

```
第 1 阶段：SFT
├─ 标注指令数据
├─ 训练模型学会跟随指令
└─ 得到 SFT 模型

第 2 阶段：训练奖励模型（Reward Modeling）
├─ 标注偏好对（哪个回答更好）
├─ 训练奖励模型：输入回答 → 输出分数
└─ 得到 RM

第 3 阶段：强化学习优化
├─ 用 RM 为生成的回答打分
├─ 梯度上升：最大化奖励
└─ 得到 RLHF 模型
```

### 2. 传统 RLHF 的三阶段详解

#### 第一阶段（已完成）

SFT 模型已经能听指令了。

#### 第二阶段：训练奖励模型

人类标注员看两条回答，选出更好的。标注数据格式：

```json
{
  "prompt": "什么是大语言模型？",
  "chosen": "LLM 是用大规模文本数据训练的神经网络…",
  "rejected": "大语言模型…（冗长、不精确或有害）"
}
```

奖励模型是一个分类器：

```python
class RewardModel(nn.Module):
    def __init__(self, base_model_path):
        super().__init__()
        self.base_model = MiniMind.from_pretrained(base_model_path)
        # 在最后一层 hidden state 上加分类头
        self.reward_head = nn.Linear(config.dim, 1)
    
    def forward(self, input_ids):
        outputs = self.base_model(input_ids, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # [B, T, dim]
        
        # 取最后一个 token 的隐藏态进行评分
        last_token_hidden = hidden_states[:, -1, :]  # [B, dim]
        reward = self.reward_head(last_token_hidden)  # [B, 1]
        
        return reward

# 训练 RM
rm_model = RewardModel(sft_model_path)
chosen_pairs = load_preference_dataset()

for chosen_text, rejected_text in chosen_pairs:
    chosen_logits = rm_model(tokenize(chosen_text))
    rejected_logits = rm_model(tokenize(rejected_text))
    
    # 奖励差：chosen 应该评分高，rejected 评分低
    loss = -torch.log(torch.sigmoid(chosen_logits - rejected_logits))
    loss.backward()
```

#### 第三阶段：强化学习（PPO）

用 RM 的奖励作为信号，优化策略（语言模型）：

$$J(\theta) = \mathbb{E}_{\pi_{\theta}}[\text{RM}(y) - \beta \cdot \text{KL}(\pi_{\theta} \| \pi_0)]$$

其中：
- $\pi_{\theta}$ 是当前策略（SFT 模型）
- $\text{RM}(y)$ 是奖励模型的得分
- $\beta \cdot \text{KL}$ 是 KL 惩罚，防止策略偏离 SFT 模型太远

PPO（Proximal Policy Optimization）是一个经典的强化学习算法，细节复杂，省略。

### 3. DPO：绕过奖励模型的巧妙方法

2023 年 Stanford 的一篇论文提出了 DPO（Direct Preference Optimization），颠覆了 RLHF 的工作流。

**核心洞察：** 不需要显式训练奖励模型！直接从偏好数据学习。

####  DPO 的数学推导（高能预警！）

假设奖励和偏好之间的关系为（Bradley-Terry 模型）：

$$P(y_1 \succ y_2 | x) = \text{sigmoid}(\text{RM}(x, y_1) - \text{RM}(x, y_2))$$

（y_1 比 y_2 更好的概率，取决于两者奖励的差）

通过一系列数学变换（拉格朗日乘数法），可以证明最优的策略满足：

$$\log \frac{\pi^*(y_1 | x)}{\pi_{\text{ref}}(y_1 | x)} - \log \frac{\pi^*(y_2 | x)}{\pi_{\text{ref}}(y_2 | x)} = \text{RM}(x, y_1) - \text{RM}(x, y_2)$$

反过来说，如果我们知道偏好对 $(y_1 \succ y_2)$，可以直接优化：

$$L_{\text{DPO}} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_1 | x)}{\pi_{\text{ref}}(y_1 | x)} - \beta \log \frac{\pi_\theta(y_2 | x)}{\pi_{\text{ref}}(y_2 | x)}\right)$$

其中 $\pi_{\text{ref}}$ 是 SFT 模型，$\beta$ 是温度参数。

### 4. DPO 的实现

```python
def compute_dpo_loss(model, ref_model, batch, beta=0.5):
    """
    model: 要训练的模型
    ref_model: 参考模型（SFT 模型，冻结不更新）
    batch: {'prompt', 'chosen', 'rejected'}
    beta: 温度参数，通常 0.5-1.0
    """
    
    # ① 计算两个回答的 log 概率
    def get_log_prob(model, prompt, response):
        # 拼接成完整输入
        full_text = prompt + response
        token_ids = tokenizer.encode(full_text)
        
        outputs = model(token_ids)
        logits = outputs.logits[:-1, :]  # 除去最后一个位置
        labels = token_ids[1:]  # 预测目标
        
        # 计算这个序列的 log 概率
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs[range(len(labels)), labels]
        return selected_log_probs.sum()
    
    # ② 计算 chosen 和 rejected 的对数概率
    chosen_log_prob = get_log_prob(model, batch['prompt'], batch['chosen'])
    rejected_log_prob = get_log_prob(model, batch['prompt'], batch['rejected'])
    
    # 参考模型的概率（用于正则化）
    with torch.no_grad():
        chosen_ref_log_prob = get_log_prob(ref_model, batch['prompt'], batch['chosen'])
        rejected_ref_log_prob = get_log_prob(ref_model, batch['prompt'], batch['rejected'])
    
    # ③ 计算 DPO 损失
    # 理想情况：chosen 远小于 rejected（log prob 的差越大越好）
    # 奖励差 = beta * (当前模型的差 - 参考模型的差)
    reward_diff = beta * (
        (chosen_log_prob - rejected_log_prob) - 
        (chosen_ref_log_prob - rejected_ref_log_prob)
    )
    
    # 用 sigmoid 交叉熵让 reward_diff > 0
    loss = -torch.nn.functional.logsigmoid(reward_diff)
    
    return loss

def train_dpo(model, ref_model, train_loader, optimizer, num_epochs=3):
    model.train()
    ref_model.eval()  # 参考模型不训练
    
    for epoch in range(num_epochs):
        for batch in train_loader:
            loss = compute_dpo_loss(model, ref_model, batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
```

**为什么 DPO 更简洁？**

| 特性 | RLHF | DPO |
|------|------|-----|
| 阶段数 | 3 阶段 | 1 阶段（相对 RM） |
| 奖励模型 | 需要显式训练 | 隐式结合在 loss 中 |
| 计算复杂度 | PPO 算法复杂 | 普通的有监督学习 |
| 实现难度 | 😰 | 😌 |
| 效果 | 最好 | 与 RLHF 相当或更好 |

---

## 💡 本节小结

| 概念 | 一句话总结 |
|------|-----------|
| RLHF | 用人类偏好作为信号，通过强化学习对齐模型与人类价值观 |
| 奖励模型 | 学习人类的评价标准，给任意回答打分 |
| PPO | 经典强化学习算法，在最大化奖励的同时保持稳定 |
| DPO | 直接优化偏好对，无需显式训练 RM，更简洁高效 |
| Bradley-Terry | 将偏好关系建模为奖励差的 sigmoid 函数 |
| beta 参数 | 控制对参考模型的偏离程度，beta 越大越激进 |

---

## 🧪 习题集 14

**题目 1（RLHF 三阶段）：** 说明 RLHF 为什么需要三个阶段，不能跳过其中任何一个？各阶段的数据形式和标注成本分别是什么？

**题目 2（DPO vs RM）：** DPO 相比 RLHF+RM+PPO，少了 RM 的显式训练。这是怎么做到的（从数学角度）？

**题目 3（log 概率计算）：** 给定模型输出和文本序列，如何计算 "这句话" 的对数概率（log probability）？写出伪代码。

**题目 4（Beta 的影响）：** 在 DPO 中，$\beta$ 是温度参数。$\beta \to 0$ 和 $\beta \to \infty$ 分别意味着什么行为？

**题目 5（设计题）：** 如果没有人类标注的偏好数据，只有 SFT 数据和一个性能指标（如回答的长度、回答的确定性），能用 DPO 吗？怎么做？

---

> 下一课我们将学习 **L15 - RLAIF——PPO / GRPO / CISPO**，以及如何用 AI 反馈获取偏好。
