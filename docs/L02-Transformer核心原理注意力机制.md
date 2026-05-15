# L02 - Transformer 核心原理——注意力机制

> **"有些事情，你必须同时看见所有，才能真正理解。"**

---

## 📌 本节目标

1. 理解 RNN 的局限性，以及为什么 Transformer 能够取代它
2. 掌握 Self-Attention 机制的直觉与公式推导
3. 理解 Q/K/V 三个矩阵分别代表什么
4. 明白为什么 Scaled Dot-Product 要除以 $\sqrt{d_k}$
5. 理解 Multi-Head Attention 和 Causal Mask
6. 结合 minimind 代码，看懂注意力计算的实现

## 📚 前置知识

- 矩阵乘法的基本概念（A × B 是什么意思）
- 知道什么是 softmax（把一堆数变成概率分布）
- 完成了 L01，理解 Token 的概念

---

## 正文

### 1. RNN 的问题：只能看前方的赛车手

在 Transformer 出现之前，处理序列数据的主角是 **RNN（循环神经网络）**。

想象一位赛车手在山路上驾驶，规则很奇特：他只能看到**正前方五米**的路，看完立刻"忘掉"它，然后把"刚才看到了什么"压缩成一个小纸条，带着这张纸条继续往前开。

这就是 RNN 处理语言的方式：
- 读入第 1 个词 → 更新"记忆向量"
- 读入第 2 个词 → 再次更新"记忆向量"
- ……
- 读入第 N 个词 → 当前的"记忆向量"要代表对整个句子的理解

**问题在于：** 如果句子很长，比如 1000 个词，那么第 1 个词的信息要通过 999 次压缩才能影响最后的输出。信息在这漫长的传递中**严重衰减**——这就是"梯度消失"问题。

类比到赛车手：山路开了三百公里，他手里那张纸条已经被折叠了两万次，上面的字迹几乎看不见了。

另一个问题：**RNN 必须串行计算**，第 i 个词的处理必须等第 i-1 个词处理完。这让 GPU 的并行计算能力完全浪费了。

---

### 2. Transformer 的灵感：鸟瞰视角的导航员

换一个场景：现在这位导航员不是在赛车手旁边，而是坐在直升机上**俯瞰整条赛道**。他可以同时看到所有的弯道、加油站、事故点，根据全局信息给出"第 37 公里处左转减速"的精确建议。

这就是 Transformer 的注意力机制：**对于序列中的每一个位置，都去"关注"（attend to）序列中的所有位置，而不仅仅是相邻的几个**。

Transformer 能并行处理所有位置——它给 GPU 带来了真正的用武之地。

---

### 3. Self-Attention 公式：图书馆检索

先给出最核心的公式，然后逐步解释每个部分：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

**类比：想象一个智慧图书馆**

你走进一座图书馆，想找关于"量子力学"的书。

- **Q（Query，查询）**：你提出的问题——"我想找量子力学的资料"
- **K（Key，键/标签）**：每本书的书签标签——"量子物理"、"核聚变"、"相对论"……
- **V（Value，值/内容）**：每本书的实际内容

你的问题 Q 会与每本书的标签 K 进行匹配，计算相似度（通过点积 $QK^T$）。相似度高的书得到更高的"关注权重"（softmax 归一化），最终把所有书的内容 V 按权重加权求和，得到一个"综合答案"。

**用数学语言重新表述：**

1. 计算 Q 与所有 K 的相似度得分：$\text{scores} = QK^T$
2. 缩放防止梯度消失（后面解释）：$\text{scores} = \frac{QK^T}{\sqrt{d_k}}$
3. Softmax 归一化成概率（注意力权重）：$\text{attn\_weights} = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$
4. 用注意力权重对 V 加权求和：$\text{output} = \text{attn\_weights} \times V$

直觉上的意义：当模型处理"苹果公司发布了新产品"中的"苹果"这个词时，注意力机制帮助它"看向""公司"这个标注，从而判断这里的"苹果"是企业名而非水果。

---

### 4. Q/K/V 是从哪里来的？

在 Self-Attention（自注意力）中，Q、K、V 都来自**同一个输入序列 X**，通过三个不同的线性变换矩阵产生：

$$
Q = XW^Q, \quad K = XW^K, \quad V = XW^V
$$

其中 $W^Q, W^K, W^V$ 是可学习的权重矩阵，维度均为 $[d_{model}, d_k]$（或 $d_v$）。

理解这个的关键：**同一个词在不同角色下有不同的表示**。

- 当作为"提问者"时，它生成 Q："我想了解什么？"
- 当作为"被查询对象"时，它生成 K："我能提供什么信息？"
- 当真正被调用时，它生成 V："这是我的实际内容"

这种设计让模型可以灵活地在"提问"和"回答"之间切换角色。

---

### 5. 为什么要除以 $\sqrt{d_k}$？

这是一个小细节，但很重要。

当 $d_k$（Key 的维度）很大时，$Q$ 与 $K$ 的点积 $QK^T$ 的数值范围会非常大（量级约为 $d_k$）。

举个例子：如果 $d_k = 64$，两个随机向量的点积的标准差约为 $\sqrt{64} = 8$。标准差越大，softmax 的输入值越极端，梯度就越小（软饱和问题）。

$$
\text{问题示意：若 z = 100 vs z = -100，softmax 几乎是 [1, 0]，梯度趋近 0}
$$

除以 $\sqrt{d_k}$ 可以把方差归一化，让 softmax 的输入保持在合理范围，梯度流动更健康。

用类比理解：把 100 个人的成绩加总再求平均，需要除以 $\sqrt{100}=10$ 来标准化——这确保不管班级大小，分数的可比性是一致的。

---

### 6. Multi-Head Attention：多角度观察

单头注意力就像用一种颜色的荧光笔标注文章——只能关注一类关系。

**Multi-Head Attention** 就是同时用多支不同颜色的荧光笔：
- 一支关注**语法结构**（"苹果"和"公司"之间的修饰关系）
- 一支关注**语义相似性**（"快乐"和"幸福"很接近）
- 一支关注**指代关系**（"他"指的是前文的哪个人）

数学上，用 $h$ 个独立的 Q/K/V 投影矩阵并行计算 $h$ 个注意力头，然后拼接起来：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

minimind-3 使用：
- `num_heads = 8`（查询头数 Q）
- `num_kv_heads = 4`（键值头数 K/V）

**Q 头数 > KV 头数** 叫做 **GQA（分组查询注意力，Grouped Query Attention）**。它让多个 Q 头共享同一对 K/V 头，减少了 KV 缓存占用，推理更快。这是 LLaMA-2、Mistral 等现代模型广泛采用的技巧。

---

### 7. Causal Mask：语言模型不能"作弊"看未来

在训练语言模型时，我们输入一个句子，让模型预测每个位置的下一个词：

```
输入：["我", "今天", "很", "开心"]
目标：["今天", "很", "开心", "<eos>"]
```

问题来了：如果模型在预测"今天"时，能看到后面的"很"和"开心"，那不就是作弊吗？

**Causal Mask（因果掩码）**就是解决这个问题的。它在计算注意力时，把"未来"的位置全部置为 $-\infty$（经过 softmax 后变成 0）：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

其中掩码矩阵 $M$ 为：

$$
M_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}
$$

直觉：位置 $i$ 的 Token 只能关注自己和它之前的所有 Token——这模拟了"自左向右生成"的过程。

---

### 8. minimind 代码片段：注意力实现

以下是 minimind `model/model_minimind.py` 中 Attention 模块的关键逻辑（简化注释版）：

```python
class Attention(nn.Module):
    def __init__(self, args: MiniMindConfig):
        super().__init__()
        # GQA：Q 头数可以多于 KV 头数
        self.n_heads = args.num_heads          # Q 的头数，例如 8
        self.n_kv_heads = args.num_kv_heads    # K/V 的头数，例如 4
        self.head_dim = args.hidden_size // args.num_heads  # 每个头的维度

        # 三个投影矩阵：将输入 x 投影到 Q、K、V 空间
        self.wq = nn.Linear(args.hidden_size, args.num_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.hidden_size, args.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.hidden_size, args.n_kv_heads * self.head_dim, bias=False)
        # 输出投影：把多头结果合并后投影回 hidden_size
        self.wo = nn.Linear(args.num_heads * self.head_dim, args.hidden_size, bias=False)

    def forward(self, x, pos_cis, mask=None, past_kv=None):
        bsz, seq_len, _ = x.shape

        # Step 1: 计算 Q、K、V
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # Step 2: reshape 为多头形状 [batch, seq, n_heads, head_dim]
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        # Step 3: 应用 RoPE 旋转位置编码（L06 详细讲）
        xq, xk = apply_rotary_emb(xq, xk, pos_cis)

        # Step 4: KV-Cache 支持（L09 详细讲）
        if past_kv is not None:
            xk = torch.cat([past_kv[0], xk], dim=1)
            xv = torch.cat([past_kv[1], xv], dim=1)

        # Step 5: GQA - 重复 KV 头以匹配 Q 头数
        xk = repeat_kv(xk, self.n_heads // self.n_kv_heads)
        xv = repeat_kv(xv, self.n_heads // self.n_kv_heads)

        # Step 6: 转置为 [batch, n_heads, seq, head_dim] 方便矩阵乘法
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Step 7: 计算注意力分数，scaled dot-product
        # scores shape: [batch, n_heads, seq_q, seq_k]
        scale = self.head_dim ** -0.5  # 除以 sqrt(d_k)
        scores = torch.matmul(xq, xk.transpose(2, 3)) * scale

        # Step 8: 应用 Causal Mask（防止看到未来）
        if mask is not None:
            scores = scores + mask  # mask 在未来位置为 -inf

        # Step 9: softmax 归一化
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)

        # Step 10: 用注意力权重加权 V，得到输出
        output = torch.matmul(scores, xv)

        # Step 11: 合并多头，投影输出
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.wo(output), (xk, xv)  # 返回输出和 KV 缓存
```

**读代码三步法：**

1. 先看 `__init__`：理解有哪些线性层（`wq`, `wk`, `wv`, `wo`）
2. 再看 `forward`：对照公式 $\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$ 找到每一步
3. 最后注意"GQA"和"KV-Cache"部分，理解现代优化技巧

---

### 9. 完整的 Transformer Block 结构

一个完整的 Transformer Block（最小重复单元）包含：

```
输入 x
  │
  ├─ LayerNorm（归一化）
  ├─ Multi-Head Self-Attention
  └─ 残差连接（x = x + attention_output）
  │
  ├─ LayerNorm（归一化）
  ├─ Feed-Forward Network (FFN / MLP)
  └─ 残差连接（x = x + ffn_output）
  │
输出 x
```

minimind-3 堆叠了 8 个这样的 Block（`num_hidden_layers=8`），每一层都重复这个结构。**堆叠越深，模型能学习的抽象层次越高**。

---

## 💡 本节小结

| 概念 | 一句话总结 |
|------|-----------|
| RNN 的问题 | 序列太长时信息衰减，且无法并行 |
| Self-Attention | 每个位置同时关注所有其他位置，计算全局依赖 |
| Q/K/V | Query=问题，Key=标签，Value=内容；均来自输入的线性变换 |
| 除以 $\sqrt{d_k}$ | 防止点积数值过大导致 softmax 饱和 |
| Multi-Head | 从多个"角度"并行关注，然后合并 |
| Causal Mask | 预测时不能看到未来的 Token |
| GQA | Q 头多于 KV 头，节省显存 |

---

## 🏋️ 习题集

**基础题：**

1. 用"图书馆查书"的类比解释 Query、Key、Value 各自的含义。
2. 在 Self-Attention 公式 $\text{softmax}(QK^T/\sqrt{d_k})V$ 中，如果 $d_k=64$，为什么要除以 8 而不是 64？
3. Causal Mask 中，$M_{ij}$ 在 $j > i$ 时为 $-\infty$，经过 softmax 后变成什么值？为什么？

**进阶题：**

1. minimind-3 有 8 个 Q 头、4 个 KV 头（GQA）。如果改成标准 MHA（Q/K/V 头数相同都是 8 头），KV-Cache 的显存占用大约会变为多少倍？
2. 假设一个句子有 1000 个 Token，Self-Attention 需要计算 $1000 \times 1000$ 的注意力矩阵。相比之下，RNN 只需要 $1000$ 步。Self-Attention 的时间复杂度是什么？这在长文本处理中意味着什么？
3. 阅读 minimind 代码中的 `apply_rotary_emb` 函数（位置编码），猜测它大致做了什么操作（L06 会详细讲解）。

---

> 下一课：**L03 - PyTorch 快速上手与数据流**
