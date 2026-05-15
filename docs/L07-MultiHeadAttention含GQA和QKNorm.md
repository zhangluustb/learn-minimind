# L07 - Multi-Head Attention（含 GQA/QK-Norm）

> **"注意力不是稀缺资源，但内存是。GQA 让我们在不牺牲智力的前提下，给计算机省钱。"**

---

## 📌 本节目标

- 深入理解标准 Multi-Head Attention（MHA）的结构与局限
- 掌握 Grouped Query Attention（GQA）的原理与工程意义
- 理解 QK-Norm 如何稳定大规模训练
- 了解 Flash Attention 的核心思想及 minimind 的调用方式
- 能读懂并解释 `model_minimind.py` 中的 `Attention` 模块

---

## 📚 前置知识

- L02 Self-Attention 基本公式
- L06 RoPE 旋转位置编码（会应用在 Q/K 上）
- PyTorch `nn.Linear`、`torch.einsum` 基础用法
- 了解 KV Cache 是什么（L09 会深入讲，本节先建立直觉）

---

## 正文

### 7.1 从标准 MHA 到 GQA（分组查询注意力）

#### 标准 Multi-Head Attention 回顾

Transformer 中最核心的部件，是 **Multi-Head Attention（MHA）**。它的思路是：与其让模型只从一个角度"看"序列，不如同时从多个子空间（头）去关注不同的语义特征。

给定输入 $X \in \mathbb{R}^{B \times L \times d}$（batch、序列长度、隐藏维度），MHA 先用三组线性变换生成 Q、K、V：

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

然后将它们拆分为 $h$ 个头，每个头独立计算 Scaled Dot-Product Attention：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中 $d_k = d / h$ 是每个头的维度。最后把所有头的结果拼接起来，经过输出投影：

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\,W^O$$

每个头的张量形状为 `[batch, seq_len, n_heads, head_dim]`。同时 K 和 V 也保持相同数量的头，因此 **KV 矩阵的头数与 Q 头数完全相同**。

---

#### MHA 的内存瓶颈：KV Cache 膨胀

在推理阶段，模型每生成一个新 token，都需要访问之前所有 token 的 K 和 V 向量（这就是 **KV Cache**）。

对标准 MHA 来说，KV Cache 的大小为：

$$\text{KV Cache} = 2 \times L \times h \times d_k \times \text{bytes\_per\_element}$$

例如，一个 `n_heads=32, head_dim=128, seq_len=4096` 的模型，用 fp16 存储：

$$2 \times 4096 \times 32 \times 128 \times 2 = 67,108,864 \text{ bytes} \approx 64 \text{ MB}$$

这只是**单个 Transformer 层**的 KV Cache。32 层叠加就是 2 GB，而这还不算模型权重本身。随着上下文变长，KV Cache 会成为部署大模型的最大内存杀手。

---

#### MQA：极端压缩的尝试

**Multi-Query Attention（MQA）** 是 2019 年 Google 提出的方案：让所有 Q 头共享同一组 K 和 V（即 `n_kv_heads=1`）。这样 KV Cache 缩减到 MHA 的 $1/h$。

但实践证明，MQA 的信息压缩过于激进——单个 K/V 头要服务所有 Q 头，模型质量下降明显，尤其在长文本推理中。

---

#### GQA：在质量与效率间找平衡

**Grouped Query Attention（GQA）** 是 2023 年 Google 提出的折中方案：

- 将 $h$ 个 Q 头分成 $g$ 组，每组 $h/g$ 个 Q 头
- 每组共享同一对 K/V 头
- 因此 KV 头数 $n_{kv\_heads} = g$，远小于 $n_{heads}$

```
Q头：  Q0  Q1  Q2  Q3  Q4  Q5  Q6  Q7
                 ↓分组（g=4）↓
KV头：  KV0     KV1     KV2     KV3
        ↑↑      ↑↑      ↑↑      ↑↑
       Q0,Q1  Q2,Q3  Q4,Q5  Q6,Q7
```

minimind 默认配置：`n_heads=8`（Q 头），`n_kv_heads=4`（KV 头），每 2 个 Q 头共享一对 KV。

| 方案 | KV 头数 | KV Cache 大小 | 模型质量 |
|------|--------|--------------|---------|
| MHA  | = Q 头数 | 基准（100%） | 最高 |
| GQA  | < Q 头数 | 压缩（例如 50%） | 接近 MHA |
| MQA  | = 1     | 极小（1/n_heads）| 明显下降 |

---

### 7.2 KV 头数 < Q 头数的工程意义

#### repeat_kv：把少数 KV 头"广播"给多数 Q 头

在实际计算中，`torch` 的 `scaled_dot_product_attention` 要求 Q、K、V 的头数一致。因此 minimind 用 `repeat_kv` 函数将 KV 头复制扩展，使其与 Q 头数量匹配：

```python
# model/model_minimind.py 中的 GQA 核心实现
class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        self.n_heads = args.n_heads           # Q 头数，如 8
        self.n_kv_heads = args.n_kv_heads     # KV 头数，如 4
        self.head_dim = args.dim // args.n_heads
        
        # 注意：k_proj 和 v_proj 输出维度是 n_kv_heads * head_dim
        self.q_proj = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.dim, args.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """将 KV 头扩展到与 Q 头数量一致"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(
        bs, slen, n_kv_heads, n_rep, head_dim
    ).reshape(bs, slen, n_kv_heads * n_rep, head_dim)
```

`n_rep = n_heads // n_kv_heads`，即每个 KV 头被复制 `n_rep` 次。这里的"复制"在张量层面只是一次 `expand`（共享内存，零拷贝），不引入额外计算开销。

---

#### 为什么 KV Cache 大小是推理瓶颈

现代 GPU（如 A100、H100）的计算能力（FLOPS）增长速度远快于内存带宽。对于 LLM 推理，每生成一个新 token，矩阵乘法量 $O(d^2)$ 相对较小，但需要从 HBM（高带宽内存）读取整个 KV Cache——这是典型的 **内存带宽受限（memory-bound）** 问题。

GQA 将 KV Cache 压缩比记为：

$$r = \frac{n_{kv\_heads}}{n_{heads}}$$

minimind 的 $r = 4/8 = 0.5$，即 KV Cache 减半。对于更大的模型（如 `n_heads=32, n_kv_heads=8`），压缩比可达 75%，推理速度提升显著。

---

### 7.3 QK-Norm 稳定训练

#### 训练不稳定的根源

在深度模型中，Q 和 K 的方差随层数加深可能变得很大。当 $\|Q\|$ 或 $\|K\|$ 过大时：

$$\frac{QK^T}{\sqrt{d_k}} \to \text{很大的绝对值}$$

这会导致 softmax 进入**饱和区**——输出趋向 one-hot，梯度接近于零，反向传播的信号几乎消失。反之，极端大的 logits 也可能引发 **梯度爆炸**。

这个问题在 **长序列**（注意力矩阵巨大）或 **大批量训练**（梯度方差大）时尤为突出。

---

#### QK-Norm 的做法

解决方案直接而有效：在计算注意力 score 之前，先对 Q 和 K 做 **归一化**。minimind 使用 RMSNorm：

$$\hat{Q} = \text{RMSNorm}(Q), \quad \hat{K} = \text{RMSNorm}(K)$$

然后用 $\hat{Q}$ 和 $\hat{K}$ 计算注意力：

$$\text{Attention}(\hat{Q}, \hat{K}, V) = \text{softmax}\left(\frac{\hat{Q}\hat{K}^T}{\sqrt{d_k}}\right)V$$

```python
# minimind 中 QK-Norm 的实现片段
class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        ...
        # qk_norm 参数控制是否开启
        self.q_norm = RMSNorm(self.head_dim) if args.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if args.qk_norm else nn.Identity()
    
    def forward(self, x, ...):
        xq = self.q_norm(xq)  # 归一化 Q
        xk = self.k_norm(xk)  # 归一化 K
        # 之后再应用 RoPE 并计算注意力
```

---

#### 为什么有 $1/\sqrt{d_k}$ 还要再加归一化？

这是一个很好的问题。$1/\sqrt{d_k}$ 的缩放只是一个**常数因子**，它能控制"期望值"，但无法限制 Q 和 K 在训练过程中的**动态范围**。

随着训练进行，Q/K 的权重可能逐渐增大，使得 $(QK^T)/\sqrt{d_k}$ 的方差持续增长。而 RMSNorm 对 Q/K **逐 token 动态归一化**，确保无论权重怎么变化，注意力分数始终处于合理范围。

两者各司其职：
- $1/\sqrt{d_k}$：**静态**缩放，匹配初始化时的期望方差
- QK-Norm：**动态**归一化，训练全程保持稳定

---

### 7.4 Flash Attention 加速原理

#### 标准 Attention 的 IO 瓶颈

标准自注意力实现需要：

1. 计算并**显式存储**注意力矩阵 $A = QK^T / \sqrt{d_k}$，形状 $[B, h, L, L]$
2. 对 $A$ 做 softmax
3. 用 $A$ 乘以 $V$

对于序列长度 $L = 4096$、`n_heads=8`，注意力矩阵大小为：

$$8 \times 4096 \times 4096 \times 2\,\text{bytes} = 256\,\text{MB}$$

这不仅占用大量 HBM，读写这块内存的 IO 成本更是计算瓶颈所在。

---

#### Flash Attention：Tiling + 在线 softmax

**Flash Attention**（Dao et al., 2022）的核心思想是 **分块（Tiling）计算**：

- 将 Q、K、V 切分为小块，逐块加载到 SRAM（片上缓存，速度远快于 HBM）
- 利用**在线 softmax（online softmax）** 技术，在不知道全局最大值的情况下增量更新结果
- 最终输出与标准 Attention 数学等价，但**从不在 HBM 中存储完整的 $[L \times L]$ 矩阵**

内存复杂度从 $O(L^2)$ 降至 $O(L)$，HBM 读写次数大幅减少，成为 **compute-bound** 而非 **memory-bound**。

---

#### minimind 的使用方式

PyTorch 2.0+ 已将 Flash Attention 集成进 `F.scaled_dot_product_attention`，minimind 直接调用：

```python
# Flash Attention with causal mask
output = F.scaled_dot_product_attention(
    xq, xk, xv,
    attn_mask=None,
    dropout_p=self.dropout if self.training else 0.0,
    is_causal=True  # 自动加 causal mask，效率比手动更高
)
```

`is_causal=True` 会自动生成下三角掩码（因果掩码），使语言模型只能看到当前 token 之前的内容，避免"未来信息泄漏"。相比手动创建 mask 矩阵，这种方式更高效、更简洁。

| 实现方式 | 内存 | 速度 | 额外 mask 矩阵 |
|---------|------|------|--------------|
| 标准 Attention | $O(L^2)$ | 慢（IO bound）| 需要 |
| Flash Attention | $O(L)$ | 快（compute bound）| 不需要 |

---

## 💡 本节小结

| 概念 | 一句话总结 |
|------|-----------|
| Multi-Head Attention | 多头并行关注不同语义子空间，拼接后线性投影输出 |
| MQA | 所有 Q 头共享 1 个 KV，极端省内存但质量有损 |
| GQA | Q 头分组共享 KV，在质量与效率间取得平衡 |
| repeat_kv | 用 expand 将 KV 广播到与 Q 头数一致，零内存开销 |
| KV Cache 瓶颈 | 推理时内存带宽是瓶颈，减少 KV 头数直接缩减传输量 |
| QK-Norm | 对 Q/K 做 RMSNorm，动态控制注意力分数范围，稳定训练 |
| $1/\sqrt{d_k}$ vs QK-Norm | 前者静态缩放初始方差，后者动态保持训练全程稳定 |
| Flash Attention | 分块计算避免存储 $O(L^2)$ 矩阵，减少 HBM IO，速度更快 |

---

## 🧪 习题集 7

### 题目 1：手动计算 Attention（基础）

给定以下小型 Attention 输入（单头，无 batch）：

$$Q = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, \quad K = \begin{bmatrix} 1 & 0 \\ 1 & 1 \end{bmatrix}, \quad V = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$$

$d_k = 2$，请计算：

1. $QK^T$
2. $QK^T / \sqrt{d_k}$（$\sqrt{2} \approx 1.414$）
3. 对每一行做 softmax（保留 2 位小数）
4. 最终输出 $\text{softmax}(\cdot) \cdot V$

> 提示：softmax 公式 $\sigma(z_i) = e^{z_i} / \sum_j e^{z_j}$，$e^0 \approx 1$，$e^{0.707} \approx 2.028$

---

### 题目 2：GQA 参数量计算（理解）

一个 minimind 模型配置如下：

- `dim = 512`（隐藏维度）
- `n_heads = 8`（Q 头数）
- `n_kv_heads = 4`（KV 头数）
- `head_dim = dim // n_heads = 64`

请计算：

1. `q_proj` 的参数量（`nn.Linear(dim, n_heads * head_dim, bias=False)`）
2. `k_proj` 的参数量（`nn.Linear(dim, n_kv_heads * head_dim, bias=False)`）
3. `v_proj` 的参数量
4. 如果改为标准 MHA（`n_kv_heads = n_heads = 8`），`k_proj` 和 `v_proj` 各增加多少参数？
5. GQA 相对 MHA，在 projection 层节省了多少百分比的参数？

---

### 题目 3：KV Cache 大小估算（工程）

假设你在部署一个拥有以下参数的模型：

- 模型层数 `n_layers = 8`
- `n_kv_heads = 4`，`head_dim = 64`
- 使用 float16（每个元素 2 字节）
- 最大序列长度 `max_seq_len = 2048`

请计算：

1. 单层、单 token 的 KV Cache 大小（字节）
2. 单层、2048 tokens 的 KV Cache 大小（KB）
3. 8 层模型、2048 tokens 的总 KV Cache（MB）
4. 如果用标准 MHA（`n_kv_heads = 8`），总 KV Cache 变为多少？压缩比是多少？

---

### 题目 4：代码理解题（进阶）

阅读以下 `repeat_kv` 函数：

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(
        bs, slen, n_kv_heads, n_rep, head_dim
    ).reshape(bs, slen, n_kv_heads * n_rep, head_dim)
```

回答以下问题：

1. 当 `n_heads=8, n_kv_heads=4` 时，`n_rep` 的值是多少？
2. 输入 `x` 的形状是什么？输出形状是什么？
3. `x[:, :, :, None, :]` 这一步的作用是什么？它将 `x` 的形状变成了什么？
4. `expand` 操作是否真正复制了数据？这对内存有什么影响？
5. 为什么最后需要 `reshape`？

---

### 题目 5：设计题（综合）

你正在设计一个用于移动端部署的轻量级语言模型，约束条件如下：

- 总参数量预算：50M
- 最大序列长度：512
- KV Cache 预算：不超过 8 MB（float16）
- 期望保持较好的模型质量

请回答：

1. 基于 KV Cache 预算，计算允许的最大 `n_kv_heads × head_dim` 乘积（假设 8 层）
2. 如果确定 `head_dim = 64`，`n_kv_heads` 最大可以是多少？
3. 你会选择 `n_heads` 为多少？理由是什么（考虑 GQA 效果和参数量）？
4. 在 QK-Norm 方面，移动端部署时你会开启还是关闭这个选项？为什么？
5. Flash Attention 在移动端（通常使用 CPU 或低端 GPU）是否同样有效？请解释。

---

> 下一课我们将学习 **L08 - FFN、SwiGLU 与 MoE**，深入解析 Transformer 中的前馈网络是如何从经典 ReLU 演进到 SwiGLU，以及 MoE（混合专家）如何用"稀疏激活"实现超大模型的高效计算。
