# L06 - Embedding 与旋转位置编码（RoPE）

> **"语言的本质是位置与语义的双重编码；告诉模型'这个词是什么'，也要告诉它'这个词在哪里'。"**

---

## 📌 本节目标

- 理解词向量 Embedding 的作用与数学本质
- 掌握绝对、相对两类位置编码的区别与取舍
- 深入理解 RoPE（旋转位置编码）的数学原理与复数实现
- 阅读 minimind 源码中的 `precompute_freqs_cis` 与 `apply_rotary_emb`
- 了解 YaRN 长文本外推技巧及其在 minimind 中的应用

## 📚 前置知识

- 线性代数基础：矩阵乘法、向量内积
- 复数基础：欧拉公式 $e^{i\theta} = \cos\theta + i\sin\theta$
- L02 中的 Self-Attention 计算流程（QKV 的来历）
- Python / PyTorch 基础操作（张量 reshape、unbind）

---

## 正文

### 6.1 词向量 Embedding

#### 从 One-Hot 到稠密向量

在将自然语言送入神经网络之前，我们必须把文字转换成数字。最朴素的方式是 **One-Hot 编码**：词表大小为 $V$，每个词用一个长度为 $V$ 的向量表示，该词对应位置为 1，其余全为 0。

这种方式有两个致命缺点：

1. **维度爆炸**：minimind 词表大小 $V = 6400$，每个词就是一个 6400 维的稀疏向量，计算和存储都极其低效。
2. **语义缺失**：所有词之间的余弦相似度都是 0，"猫"和"猫咪"在 One-Hot 空间中和"飞机"与"香蕉"一样"不相关"。

**词嵌入（Word Embedding）** 解决了上述问题——用一个可学习的映射矩阵 $W_E \in \mathbb{R}^{V \times d}$，将每个 token id 映射为一个 $d$ 维稠密向量：

$$\mathbf{e}_i = W_E[i] \quad \in \mathbb{R}^d$$

这等价于一次矩阵查表操作（lookup），也可以理解为用 One-Hot 向量 $\mathbf{v}_i$ 左乘权重矩阵：

$$\mathbf{e}_i = \mathbf{v}_i \cdot W_E$$

由于 $\mathbf{v}_i$ 只有第 $i$ 维为 1，乘法结果恰好就是 $W_E$ 的第 $i$ 行，即一次 **O(1)** 的行索引操作。

#### 语义近似 = 向量近似

训练完成后，语义相近的词会聚集在向量空间的相似区域。经典的例子：

$$\text{vec}(\text{国王}) - \text{vec}(\text{男人}) + \text{vec}(\text{女人}) \approx \text{vec}(\text{女王})$$

这种"语义算数"能力并非显式编程，而是模型在海量语料上反向传播后自然涌现的。

#### minimind 中的 Embedding 层

minimind 的词表大小为 6400，默认 `hidden_size=512`，因此 Embedding 矩阵形状为 $6400 \times 512$，参数量约 **3.28M**——占整个小模型参数量的相当一部分。

```python
# model/model_minimind.py
class MiniMindModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 词嵌入层：vocab_size × hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        # ... 其他层
```

输入的 token id 序列形状为 `(batch, seq_len)`，经过 `embed_tokens` 后变为 `(batch, seq_len, hidden_size)`，即每个位置都有一个 512 维的语义向量。

---

### 6.2 绝对位置编码 vs 相对位置编码

#### 为什么 Attention 需要位置信息

回顾 Self-Attention 的计算：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

观察可以发现，$Q$ 和 $K$ 的内积是**位置无关**的——无论"我爱你"还是"你爱我"，纯粹的 Attention 计算会产生完全相同的输出，因为它只关心"哪些词出现了"，不关心"词的顺序"。这显然不符合语言的实际规律。

位置编码的任务就是**将位置信息注入模型**，让模型能区分第 1 个词和第 5 个词。

#### 绝对位置编码：原版 Transformer 的 Sinusoidal PE

2017 年原版 Transformer 论文提出了正弦/余弦位置编码：

$$PE_{(pos,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

$$PE_{(pos,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

其中 $pos$ 是 token 在序列中的位置，$i$ 是向量维度的索引（从 0 到 $d/2 - 1$）。

| 特性 | 说明 |
|------|------|
| 不需要训练 | 公式确定，不占用参数量 |
| 可以外推 | 理论上支持任意长度序列 |
| 语义模糊 | 位置信息与语义向量简单相加，交互不充分 |
| 泛化受限 | 实践中长序列性能仍然下降 |

#### 学习型位置编码（BERT / GPT-2）

另一种思路是直接为每个位置学习一个可训练向量，训练期间模型自行决定如何编码位置：

```python
# 学习型绝对位置编码示意
self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
pos_ids = torch.arange(seq_len)
pe = self.position_embedding(pos_ids)  # (seq_len, hidden_size)
x = token_embedding + pe
```

优点是简单灵活，缺点是**无法外推**——训练时见过的最大位置是多少，推理时就只能用到多长的序列。

#### 相对位置编码的动机

绝对位置编码的核心假设是：位置 3 的词和位置 7 的词之间的关系，可以由"绝对位置 3"和"绝对位置 7"的编码决定。

但实际上语言学中更重要的是**相对距离**——"今天吃了"和"他今天吃了"中，"吃"和"今天"的关系是一样的，不管它们的绝对位置是多少。

相对位置编码的目标：让 $q_m^T k_n$ 这个注意力分数**只依赖于相对距离 $m - n$**，而非绝对位置 $m$ 和 $n$。RoPE 就是实现这一目标的优雅方案。

---

### 6.3 RoPE 数学原理与代码实现

#### 核心思想：用旋转编码相对位置

RoPE（Rotary Position Embedding，Su et al. 2021）的关键洞察是：

> **在 Q 和 K 上分别施加与位置相关的旋转变换，使得它们内积只反映相对位置差。**

形式化表述：设位置 $m$ 处的查询向量为 $\mathbf{q}$，位置 $n$ 处的键向量为 $\mathbf{k}$，RoPE 定义：

$$f_q(\mathbf{x}_m, m) = R_m \mathbf{q}, \quad f_k(\mathbf{x}_n, n) = R_n \mathbf{k}$$

其中 $R_\theta$ 是旋转矩阵，旋转角度与位置 $\theta$ 成正比。两者的内积为：

$$\mathbf{q}_m^T \mathbf{k}_n = (R_m \mathbf{q})^T (R_n \mathbf{k}) = \mathbf{q}^T R_m^T R_n \mathbf{k} = \mathbf{q}^T R_{n-m} \mathbf{k}$$

**内积只依赖于相对距离 $n - m$**，这正是我们想要的！

#### 二维情形的旋转矩阵

对于二维向量 $(x_1, x_2)$，旋转角 $\theta$ 对应的旋转矩阵为：

$$R_\theta = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix}$$

对 $d$ 维向量，RoPE 将其两两分组，每对维度 $(2i, 2i+1)$ 使用独立的旋转频率 $\theta_i$：

$$\theta_i = \frac{1}{10000^{2i/d}}$$

这和 Sinusoidal PE 的频率设计一脉相承，但作用方式从"相加"变成了"旋转"——更有效地编码了位置间的相对关系。

#### 复数乘法：旋转的高效表示

在实现层面，二维旋转可以用复数乘法代替矩阵乘法，效率更高：

$$\text{旋转}(x_1 + ix_2, \theta) = (x_1 + ix_2) \cdot e^{i\theta} = (x_1\cos\theta - x_2\sin\theta) + i(x_1\sin\theta + x_2\cos\theta)$$

minimind 正是利用了这一点，将 $\cos$ 和 $\sin$ 预计算好，然后通过实数乘法高效模拟复数旋转：

```python
# model/model_minimind.py 中的 RoPE 实现

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """预计算旋转位置编码的复数表示"""
    # 计算每对维度的旋转频率：θ_i = 1 / 10000^(2i/dim)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    # 生成位置序列 [0, 1, 2, ..., end-1]
    t = torch.arange(end, device=freqs.device)
    # 外积：每个位置乘以每个频率 → shape: (end, dim/2)
    freqs = torch.outer(t, freqs).float()
    # 转为复数形式 e^(i*freq)：分别存 cos 和 sin
    freqs_cos = torch.cos(freqs)  # (end, dim/2)
    freqs_sin = torch.sin(freqs)  # (end, dim/2)
    return freqs_cos, freqs_sin


def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    """应用旋转位置编码到 Q 和 K"""
    # 将最后一维拆分为复数形式：(..., dim) → (..., dim/2, 2)
    xq_r, xq_i = xq.float().reshape(*xq.shape[:-1], -1, 2).unbind(-1)
    xk_r, xk_i = xk.float().reshape(*xk.shape[:-1], -1, 2).unbind(-1)

    # 复数旋转：(a+bi)(cosθ+i·sinθ) = (a·cosθ - b·sinθ) + i(a·sinθ + b·cosθ)
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # 将实部和虚部重新拼接，恢复原始形状
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(-2)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(-2)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

**逐行解析：**

- `freqs = 1.0 / (theta ** ...)` — 计算 $\theta_i = 10000^{-2i/d}$，共 $d/2$ 个频率
- `torch.outer(t, freqs)` — 外积得到 `(seq_len, d/2)` 的角度矩阵，位置 $m$、维度 $i$ 的角度为 $m \cdot \theta_i$
- `unbind(-1)` — 沿最后一维解绑，相当于将每对 $(x_{2i}, x_{2i+1})$ 分离为实部和虚部
- 最终 `xq_out` 形状与输入 `xq` 相同，只是每个值都已经过旋转

#### RoPE 的优势对比

| 维度 | Sinusoidal PE | 学习型绝对 PE | RoPE |
|------|--------------|--------------|------|
| 训练参数 | 无 | 有 | 无 |
| 长度外推 | 较差 | 不能外推 | 优秀 |
| 相对位置感知 | 间接 | 间接 | 直接 |
| 实现复杂度 | 低 | 低 | 中等 |
| 主流使用 | 早期 Transformer | BERT/GPT-2 | LLaMA/minimind |

---

### 6.4 YaRN——长文本外推技巧

#### 问题背景

minimind 默认训练 `max_seq_len=512`。在推理时，如果输入序列超过 512，RoPE 的旋转角度会超出训练时见过的范围，导致注意力分数分布失真，性能大幅下降。

这并非 RoPE 独有的问题——任何位置编码方案都面临**长度泛化（length generalization）**的挑战。

#### YaRN 的思路

YaRN（Yet another RoPE extensioN，Peng et al. 2023）的核心思想是：

> **对不同频率的维度采用不同的缩放策略**，让高频维度（对近距离敏感）保持不变，对低频维度（对远距离敏感）进行插值扩展。

直觉上可以理解为：时钟的秒针和时针走一圈的时间不同——秒针（高频）无需调整，时针（低频）需要"拉伸"以适应更长的时间段。

具体来说，YaRN 将旋转角度从 $m \cdot \theta_i$ 修改为：

$$m \cdot \theta_i' = m \cdot \left(\frac{\theta_i}{s}\right)$$

其中缩放系数 $s = L_{\text{target}} / L_{\text{train}}$ 是目标长度与训练长度之比，但不同频率维度的缩放比例不同（分段线性插值）。

#### minimind 中的实现

minimind 通过配置项支持 YaRN：

```python
# model/LMConfig.py（配置示意）
@dataclass
class LMConfig:
    # ...
    rope_scaling: Optional[dict] = None  # YaRN 缩放配置
    # 示例配置：{"type": "yarn", "factor": 4.0, "original_max_position_embeddings": 512}
```

当 `rope_scaling` 不为 None 时，`precompute_freqs_cis` 会在计算频率时引入缩放因子，将有效上下文窗口从 512 扩展到 `512 × factor = 2048`（甚至更长）。

#### 什么时候需要开启 YaRN

| 场景 | 建议 |
|------|------|
| 推理序列 ≤ 训练长度 | 无需开启，保持默认 |
| 推理序列为训练长度 2-4 倍 | 开启 YaRN，factor 设为对应倍数 |
| 需要超长文档理解 | 结合 YaRN + 滑动窗口注意力 |
| 重新 fine-tune 到更长长度 | 优先 full fine-tune，YaRN 作为补充 |

需要注意：YaRN 是一种**免训练的近似方法**，在极长序列上不能完全替代真正的长文本训练。如果应用场景对长文本质量要求极高，最稳健的做法仍是用长文本数据重新训练或持续预训练。

---

## 💡 本节小结

| 概念 | 一句话总结 |
|------|-----------|
| Word Embedding | 可训练的查找表，将 token id 映射为稠密语义向量 |
| Sinusoidal PE | 用正余弦函数为每个位置生成固定编码，简单但外推能力弱 |
| 学习型绝对 PE | 直接学习位置向量，灵活但无法外推到训练外长度 |
| RoPE | 在 Q/K 上施加旋转变换，内积自然体现相对位置，是当前主流方案 |
| 复数旋转实现 | 通过预计算 cos/sin 并模拟复数乘法，高效无额外参数 |
| YaRN | 对不同频率维度差异化缩放，免训练地将上下文窗口外推到更长长度 |

---

## 🧪 习题集 6

**习题 1（Embedding 维度计算）**

minimind 词表大小 $V = 6400$，`hidden_size = 512`。

(a) Embedding 矩阵的参数量是多少？

(b) 若将 `hidden_size` 扩大到 1024，参数量变为多少？增加了几倍？

(c) 输入 batch size=2、seq_len=128 的 token id 张量，经过 Embedding 后，输出张量的形状是什么？

---

**习题 2（位置编码类型对比）**

下表列出了三种位置编码方案，请填写空白处：

| 方案 | 是否需要训练参数 | 能否外推到训练外长度 | 使用此方案的代表模型 |
|------|----------------|---------------------|-------------------|
| Sinusoidal PE | _______ | _______ | _______ |
| 学习型绝对 PE | _______ | _______ | _______ |
| RoPE | _______ | _______ | _______ |

并简要解释：为什么学习型绝对 PE 无法外推？

---

**习题 3（RoPE 旋转矩阵理解）**

设有二维向量 $\mathbf{q} = (1, 0)$，旋转角 $\theta = \pi/4$（45°）。

(a) 手动计算 $R_\theta \mathbf{q}$ 的结果（用分数或根号表示）。

(b) 设 $\mathbf{k} = (0, 1)$，位置差 $n - m = 1$，验证 $\mathbf{q}^T R_{-1} \mathbf{k}$ 的值（其中 $R_{-1}$ 表示旋转角为 $-\theta = -\pi/4$ 的矩阵）。

(c) 如果位置差变为 $n - m = 2$（旋转角变为 $-2\theta$），内积会如何变化？这说明 RoPE 能捕捉什么信息？

---

**习题 4（代码阅读题）**

阅读下面的代码片段：

```python
freqs = 1.0 / (10000.0 ** (torch.arange(0, 64, 2).float() / 64))
t = torch.arange(512)
freqs = torch.outer(t, freqs)
freqs_cos = torch.cos(freqs)
freqs_sin = torch.sin(freqs)
```

(a) `freqs` 在 `torch.outer` 之前的形状是什么？`torch.outer` 之后呢？

(b) `freqs_cos` 的形状是什么？它的第 $[m, i]$ 元素代表什么含义？

(c) 如果将 `64` 改为 `128`（对应 `hidden_size=128`，单头），代码哪里需要修改？

(d) 为什么代码中用 `step=2`（即 `arange(0, dim, 2)`）而不是 `arange(0, dim)`？

---

**习题 5（YaRN 设计题）**

假设你训练了一个 minimind 模型，`max_seq_len=512`，现在需要在生产环境中处理最长 2048 token 的文档。

(a) 如果不做任何调整直接推理，可能会出现什么问题？（从 RoPE 角度解释）

(b) 使用 YaRN 时，`factor` 应该设置为多少？

(c) YaRN 相比直接重新训练一个 `max_seq_len=2048` 的模型，有何优缺点？各适用什么场景？

(d)（思考题）除了 YaRN，你还能想到哪些处理"推理时序列长于训练长度"问题的方法？各有什么局限？

---

> 下一课我们将学习 **L07 - Multi-Head Attention（含 GQA/QK-Norm）**，深入研究注意力头的分组策略与训练稳定性技巧。
