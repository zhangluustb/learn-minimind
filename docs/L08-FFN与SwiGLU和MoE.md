# L08 - FFN、SwiGLU 与 MoE

> **"Attention 负责找关系，FFN 负责做推理——前者是侦探，后者是法官。"**

---

## 📌 本节目标

- 理解 FFN 在 Transformer 中的角色与"放大-激活-收缩"结构
- 掌握从 ReLU 到 SwiGLU 的激活函数演进，理解门控机制的意义
- 理解 MoE（混合专家）的稀疏激活思想、路由机制与负载均衡
- 读懂 minimind 中 `FeedForward` 与 `MOEFeedForward` 的完整源码

---

## 📚 前置知识

- L02：Transformer 整体结构，知道每个 Block 包含 Attention + FFN
- L07：Multi-Head Attention 的工作方式（维度转换、多头混合）
- 基本线性代数：矩阵乘法、维度变换
- PyTorch 基础：`nn.Linear`、`nn.ModuleList`、`F.silu`

---

## 正文

### 8.1 FFN 的"放大-激活-收缩"

#### 为什么 Transformer 需要 FFN？

在 Transformer 的每个 Block 中，Multi-Head Attention 做的事情是**跨 token 的信息混合**——它回答"哪些位置的信息与我相关"，然后把这些信息加权求和。但 Attention 本质上是一种**线性组合**，它能聚合信息，却不能对单个 token 的特征进行复杂的非线性变换。

这就是 FFN（Feed-Forward Network，前馈网络）存在的原因：**对每个 token 的特征向量分别进行非线性变换**，相当于给模型增加"推理"和"记忆"的能力。

> 类比：Attention 是"侦探"，负责收集线索、建立联系；FFN 是"法官"，对每条线索独立进行分析判断，得出结论。

#### 传统 FFN 的结构

最初 Transformer 论文（Vaswani et al., 2017）里的 FFN 非常简单：

$$\text{FFN}(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$$

其中输入维度为 $d_{\text{model}}$，中间维度（intermediate size）通常取 $4 \times d_{\text{model}}$：

```
输入 x: [batch, seq_len, dim]
  ↓ Linear(dim → 4*dim)
  ↓ ReLU 激活
  ↓ Linear(4*dim → dim)
输出: [batch, seq_len, dim]
```

**参数量分析**：

| 组件 | 参数量 |
|------|--------|
| $W_1$：$d \times 4d$ | $4d^2$ |
| $W_2$：$4d \times d$ | $4d^2$ |
| 传统 FFN 合计 | $8d^2$ |
| 对比：MHA 的 $W_Q, W_K, W_V, W_O$ | $4d^2$ |

FFN 的参数量是 Attention 的两倍！这说明 FFN 在 Transformer 中承担了大量的参数存储任务，很多研究认为 **FFN 层相当于模型的"知识库"**，事实性知识（factual knowledge）大量存储于此。

#### 放大倍数的选择

minimind 中 `hidden_dim` 的默认计算方式为：

$$\text{hidden\_dim} = \left\lfloor \frac{4d \times 2}{3} \right\rfloor$$

这不是随意的数字——它来自 SwiGLU 的参数补偿策略（见 8.2 节）。

---

### 8.2 SwiGLU 激活函数

#### 激活函数的演进

| 激活函数 | 公式 | 特点 |
|----------|------|------|
| ReLU | $\max(0, x)$ | 简单高效，但负值梯度为 0（死神经元） |
| GELU | $x \cdot \Phi(x)$ | 更平滑，NLP 表现更好；$\Phi$ 为正态分布 CDF |
| Swish | $x \cdot \sigma(x)$ | GELU 的近似，可微，无硬截断 |
| SwiGLU | $\text{Swish}(xW) \otimes xV$ | 带门控，GLU 变体，当前 LLM 首选 |

**Swish 激活函数**定义为：

$$\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

它在 $x < 0$ 时不完全截断（不像 ReLU），保留了一定的负值信息，梯度更平滑。

#### SwiGLU：门控线性单元

SwiGLU 由 Noam Shazeer 在 2020 年提出。其核心思想是引入**门控机制（Gating）**：

$$\text{FFN}_{\text{SwiGLU}}(x, W, V, W_2) = (\text{Swish}(xW) \otimes xV) \cdot W_2$$

用文字描述：
1. $xW$：生成"门控信号"，决定哪些特征值得激活
2. $\text{Swish}(xW)$：对门控信号做平滑激活
3. $xV$：生成"内容信号"，存储需要传递的信息
4. $\otimes$：逐元素乘法，门控信号决定内容信号的通过量
5. $\cdot W_2$：投影回原始维度

**直觉理解**：
- $xV$ 是"候选信息"，$\text{Swish}(xW)$ 是"阀门"
- 阀门接近 0 时，该特征被抑制；阀门接近 1 时，该特征被放行
- 这给了模型按内容动态过滤信息的能力

#### 为什么需要三个权重矩阵？

传统 FFN 用 2 个矩阵（$W_1, W_2$），SwiGLU 需要 3 个（$W_1/w_1, W_3/w_3, W_2/w_2$）：

```
w1: dim → hidden_dim   （gate 权重）
w3: dim → hidden_dim   （value 权重）
w2: hidden_dim → dim   （投影回原维度）
```

为了维持与传统 FFN **相近的总参数量**，minimind 将 `hidden_dim` 缩小为 $\frac{8d}{3}$（即 $4d \times \frac{2}{3}$）：

$$\underbrace{2 \times d \times \frac{8d}{3}}_{\text{SwiGLU 两个升维矩阵}} + \underbrace{\frac{8d}{3} \times d}_{\text{降维矩阵}} = 8d^2 \approx \text{传统 FFN 的参数量}$$

#### minimind 中的 SwiGLU 实现

```python
class FeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        # hidden_dim 通常是 dim * 8/3，取 multiple_of 的整数倍
        hidden_dim = int(config.dim * 4 * 2 / 3)
        if config.hidden_dim is not None:
            hidden_dim = config.hidden_dim
        # 取整到 multiple_of 的倍数，方便硬件对齐
        hidden_dim = config.multiple_of * (
            (hidden_dim + config.multiple_of - 1) // config.multiple_of
        )

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)  # gate 权重
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)  # 投影回原维度
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)  # value 权重
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # SwiGLU: silu(xW1) ⊗ (xW3) 然后投影回 dim
        # F.silu 即 Swish 激活：x * sigmoid(x)
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))
```

注意：`F.silu` 即 Swish 激活（PyTorch 中称为 SiLU——Sigmoid Linear Unit）。

---

### 8.3 MoE（混合专家）路由机制

#### 为什么需要 MoE？

随着模型规模的扩大，一个核心矛盾出现了：

- **更多参数**意味着更强的表达能力
- **更多参数**也意味着更大的计算开销

MoE（Mixture of Experts，混合专家）用一种优雅的方式打破这个矛盾：**增加参数量，但不增加计算量**。

其核心思想是**稀疏激活**：将 FFN 替换为 $N$ 个独立的 Expert FFN，每次前向传播时，每个 token 只激活其中的 $k$ 个 Expert（通常 $k \ll N$）。

#### Dense 与 Sparse 模型的对比

| 特性 | Dense FFN | MoE FFN |
|------|-----------|---------|
| 总参数量 | $d \times h$ | $N \times d \times h$ |
| 每次实际计算量 | 全部参数参与 | 仅 $k/N$ 比例参与 |
| 显存占用 | 低 | 高（需同时加载所有 Expert） |
| 计算效率 | 正比于参数量 | 解耦参数量与计算量 |
| 典型代表 | LLaMA、GPT-2 | Mixtral、DeepSeek、minimind |

#### Router 路由网络

MoE 的核心是一个**路由器（Router）**，它决定每个 token 由哪些 Expert 处理：

$$G(x) = \text{softmax}(\text{TopK}(x \cdot W_g))$$

具体步骤：
1. 计算路由 logits：$\text{logits} = x W_g \in \mathbb{R}^N$（$N$ 为 Expert 数量）
2. Softmax 归一化得到路由权重
3. 选取 Top-$k$ 个权重最大的 Expert
4. 对选中 Expert 的输出做加权求和：

$$\text{output}(x) = \sum_{i \in \text{TopK}} G_i(x) \cdot \text{Expert}_i(x)$$

#### 负载均衡问题

路由器容易陷入**Expert 塌陷（Expert Collapse）**：模型发现某 1-2 个 Expert 效果最好，导致所有 token 都选择同一个 Expert，其他 Expert 几乎不被训练。

解决方案：在训练时加入**辅助损失（Auxiliary Loss）**，惩罚 Expert 使用不均匀的情况：

$$\mathcal{L}_{\text{aux}} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

其中 $f_i$ 是第 $i$ 个 Expert 被选中的频率，$P_i$ 是路由器给第 $i$ 个 Expert 的平均概率，$\alpha$ 是平衡系数。

---

### 8.4 minimind MoE 源码解析（MOEFeedForward）

#### MiniMindConfig 中的 MoE 参数

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `use_moe` | 是否启用 MoE | `True/False` |
| `n_routed_experts` | 可路由 Expert 总数 | 8 |
| `num_experts_per_tok` | 每个 token 激活的 Expert 数 | 2 |
| `n_shared_experts` | 共享 Expert 数量 | 1 |

#### Shared Expert 的作用

minimind 在 MoE 中引入了**共享 Expert**（Shared Expert），这是 DeepSeek-MoE 提出的设计：

- **路由 Expert**：每个 token 选 Top-$k$ 个，不同 token 可能选不同 Expert
- **共享 Expert**：所有 token 都必须经过，处理通用信息

共享 Expert 的好处：
1. 保证所有 token 都有"基础处理"能力，防止某些 token 完全依赖稀疏路由
2. 减轻路由 Expert 的压力，路由 Expert 可以更专注于"专业信息"
3. 提升训练稳定性

#### MOEFeedForward 完整实现

```python
class MOEFeedForward(nn.Module):
    def __init__(self, config: MiniMindConfig):
        super().__init__()
        self.config = config
        # 路由网络：输入 dim，输出 n_routed_experts 个 logits
        self.gate = nn.Linear(config.dim, config.n_routed_experts, bias=False)
        # 多个独立 Expert，每个是标准 SwiGLU FFN
        self.experts = nn.ModuleList([
            FeedForward(config) for _ in range(config.n_routed_experts)
        ])
        # 共享 Expert（所有 token 都会经过）
        self.shared_experts = FeedForward(config)

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape

        # 展平 batch 和 seq 维度，让每个 token 独立路由
        # x: [bsz, seq_len, dim] → [bsz*seq_len, dim]
        x = x.view(-1, x.shape[-1])

        # 路由：为每个 token 计算 Expert 权重
        router_logits = self.gate(x)                          # [T, n_experts]
        routing_weights = F.softmax(router_logits, dim=-1)   # 归一化为概率
        routing_weights, selected_experts = torch.topk(
            routing_weights, self.config.num_experts_per_tok, dim=-1
        )   # 各取 top-k：[T, k]
        # 对选中的 k 个权重重新归一化，使之和为 1
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # 稀疏激活：逐 Expert 聚合 token
        final_hidden_states = torch.zeros_like(x)
        # expert_mask[i]: 第 i 个 Expert 负责哪些 token（以及是哪个 top-k 位置）
        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.config.n_routed_experts
        ).permute(2, 1, 0)  # [n_experts, k, T]

        for expert_idx in range(self.config.n_routed_experts):
            expert = self.experts[expert_idx]
            # 找出分配到该 Expert 的 token 索引
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue  # 该 Expert 本轮没有被选中
            # 取出对应 token，送入 Expert 计算
            current_state = x[None, top_x].reshape(-1, x.shape[-1])
            # 乘以路由权重，加权累加回结果张量
            current_state = expert(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_state)

        # 恢复形状，加上共享 Expert 的输出
        final_hidden_states = final_hidden_states.reshape(orig_shape)
        final_hidden_states = final_hidden_states + self.shared_experts(identity)
        return final_hidden_states
```

#### 数据流一览

```
输入 x: [B, T, dim]
    │
    ├─ identity = x                      # 保留原始，供 shared expert 使用
    │
    ├─ view(-1, dim) → [B*T, dim]
    │       ↓
    │   gate(x) → router_logits [B*T, N]
    │       ↓ softmax + topk
    │   selected_experts, weights [B*T, k]
    │       ↓ 稀疏激活（逐 expert 处理）
    │   expert_outputs [B*T, dim]
    │       ↓ reshape
    │   routed_output [B, T, dim]
    │
    └─ shared_experts(identity) [B, T, dim]
           ↓
    输出 = routed_output + shared_output [B, T, dim]
```

---

## 💡 本节小结

| 概念 | 一句话总结 |
|------|-----------|
| FFN 的作用 | 对每个 token 独立做非线性特征变换，是模型"知识存储"的主要位置 |
| 放大倍数 | intermediate_size ≈ 4×dim，SwiGLU 下取 8d/3 以维持参数量平衡 |
| SwiGLU | 带门控的 GLU 变体：$\text{Swish}(xW_1) \otimes xW_3$，三个权重矩阵 |
| 门控机制 | $\text{Swish}(xW_1)$ 作阀门，动态决定 $xW_3$ 的哪些特征被放行 |
| MoE 核心思想 | 稀疏激活：$N$ 个 Expert 只激活 $k$ 个，解耦参数量与计算量 |
| Router | $\text{softmax}(\text{TopK}(xW_g))$，学习 token 与 Expert 的亲和度 |
| 负载均衡 | 辅助 loss 惩罚使用不均匀，防止 Expert 塌陷 |
| Shared Expert | 所有 token 共享的通用 FFN，提升稳定性与基础表达能力 |

---

## 🧪 习题集 8

**习题 8.1：FFN 参数量计算对比**

设模型维度 $d = 512$。

1. 计算传统 FFN（$4\times$ 放大）的参数量（忽略 bias）
2. 计算 SwiGLU FFN（`hidden_dim = 8d/3`，忽略取整）的参数量
3. 两者有多少差异？SwiGLU 是否比传统 FFN 参数量更多？

> 提示：SwiGLU 有三个矩阵 $w_1, w_2, w_3$，尺寸分别为 $d \times h$、$h \times d$、$d \times h$

---

**习题 8.2：SwiGLU forward pass 手算**

给定（简化）参数：
- 输入 $x = [1, 0, -1, 2]$（维度 4）
- $W_1 = I_{4}$（单位矩阵），$W_3 = I_{4}$，$W_2 = I_{4}$
- $\text{Swish}(x) = x \cdot \sigma(x)$，$\sigma(x) = \frac{1}{1+e^{-x}}$

请手动计算 $\text{FFN}_{\text{SwiGLU}}(x)$ 的输出，并说明门控信号对各维度的影响。

---

**习题 8.3：MoE 稀疏激活率计算**

minimind 默认配置：`n_routed_experts=8`，`num_experts_per_tok=2`。

1. 计算每个 token 的稀疏激活率（即实际计算量 / 理论最大计算量）
2. 若将 Expert 数量扩展到 64，激活数仍为 2，激活率如何变化？
3. 在参数量上，8 个 Expert + 1 个 Shared Expert 的总 FFN 参数量是普通 Dense FFN 的几倍？

---

**习题 8.4：代码理解——Shared Expert 的必要性**

阅读 `MOEFeedForward.forward()` 的代码，回答以下问题：

1. 如果去掉 `self.shared_experts`，仅保留路由 Expert，会有什么问题？
2. `identity = x` 为什么要在 `x = x.view(-1, ...)` 之前保存？
3. `routing_weights / routing_weights.sum(dim=-1, keepdim=True)` 这一步归一化的目的是什么？如果不做会怎样？

---

**习题 8.5：设计题——如何防止 MoE 路由塌陷（Expert Collapse）**

Expert 塌陷是指路由器总是选择少数几个 Expert，导致其余 Expert 得不到训练。

请从以下角度分析并提出解决方案：

1. **辅助损失**：写出负载均衡 loss 的直觉解释，说明它如何引导 Expert 使用均匀化
2. **Expert 初始化**：为什么所有 Expert 不应该用完全相同的初始参数？如何打破对称性？
3. **Top-k 选择策略**：除了 softmax + TopK，还有哪些路由方式可以天然地鼓励负载均衡？（提示：考虑 noise、expert capacity 等关键词）

---

> 下一课我们将学习 **L09 - KV-Cache 与推理加速**，探索为什么生成式模型在推理时不需要重复计算，以及 KV-Cache 是如何在代码中流转的。
